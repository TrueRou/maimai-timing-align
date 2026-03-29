from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from .analysis import align_videos
from .exporter import export_aligned_video
from .media import ensure_ffmpeg_available
from .models import AlignConfig, AlignResult


def _fmt_sec(v: float) -> str:
    return f"{v:.3f}s"


def _save_uploaded(uploaded, target_dir: Path) -> Path:
    suffix = Path(uploaded.name).suffix or ".mp4"
    out = target_dir / f"{uploaded.file_id}{suffix}"
    out.write_bytes(uploaded.getbuffer())
    return out


def _collect_debug_frames(video_path: Path, debug_info, max_around: int = 20) -> list[dict]:
    if not debug_info or debug_info.selected_index is None:
        return []
    if not debug_info.source_frame_indices:
        return []

    idx = int(debug_info.selected_index)
    ts = debug_info.timestamps or []
    scores = debug_info.score or []
    src_frames = debug_info.source_frame_indices or []
    if idx < 0 or idx >= len(src_frames):
        return []

    start = max(0, idx - max_around)
    end = min(len(src_frames), idx + max_around + 1)
    target_rows = list(range(start, end))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames: list[dict] = []
    try:
        for i in target_rows:
            src = int(src_frames[i])
            cap.set(cv2.CAP_PROP_POS_FRAMES, src)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            ok_enc, buf = cv2.imencode(".jpg", frame)
            if not ok_enc:
                continue

            frames.append(
                {
                    "rel": i - idx,
                    "sample_idx": i,
                    "src_frame": src,
                    "t_sec": float(ts[i]) if i < len(ts) else None,
                    "score": float(scores[i]) if i < len(scores) else None,
                    "selected": i == idx,
                    "image": buf.tobytes(),
                }
            )
    finally:
        cap.release()

    return frames


def _show_debug(result: AlignResult, debug_frames: dict | None = None) -> None:
    st.subheader("检测诊断")

    around_frames = st.slider("调试：显示锚点前后帧数", 2, 20, 8, 1)

    def _show_anchor_window(title: str, debug_info) -> None:
        if not debug_info or debug_info.selected_index is None:
            return

        idx = int(debug_info.selected_index)
        ts = debug_info.timestamps
        scores = debug_info.score
        src_frames = debug_info.source_frame_indices
        if not ts or not scores:
            return

        if idx < 0 or idx >= len(ts):
            return

        selected_t = float(ts[idx])
        selected_src = None
        if src_frames and idx < len(src_frames):
            selected_src = int(src_frames[idx])

        label = f"{title} 实际采用时间点: {selected_t:.3f}s"
        if selected_src is not None:
            label += f"（源帧 #{selected_src}）"
        st.caption(label)

        start = max(0, idx - around_frames)
        end = min(len(ts), idx + around_frames + 1)

        rows: list[dict[str, float | int | str]] = []
        for i in range(start, end):
            row: dict[str, float | int | str] = {
                "rel": i - idx,
                "sample_idx": i,
                "t_sec": float(ts[i]),
                "score": float(scores[i]),
                "selected": "<--" if i == idx else "",
            }
            if src_frames and i < len(src_frames):
                row["src_frame"] = int(src_frames[i])
            rows.append(row)

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        if debug_frames and title in debug_frames:
            subset = [x for x in debug_frames[title] if abs(int(x["rel"])) <= around_frames]
            if subset:
                st.markdown(f"**{title} 锚点前后帧预览**")
                cols = st.columns(5)
                for j, item in enumerate(subset):
                    col = cols[j % 5]
                    caption = (
                        f"rel={item['rel']} | src={item['src_frame']}\n"
                        f"t={item['t_sec']:.3f}s | s={item['score']:.3f}"
                    )
                    if item["selected"]:
                        caption = "[SELECTED]\n" + caption
                    col.image(item["image"], caption=caption, width=180)

    if result.debug1:
        st.markdown("**Clip1 转场分数**")
        df1 = pd.DataFrame({"t": result.debug1.timestamps, "score": result.debug1.score}).set_index("t")
        st.line_chart(df1)
        _show_anchor_window("Clip1", result.debug1)
    if result.debug2:
        st.markdown("**Clip2 转场分数**")
        df2 = pd.DataFrame({"t": result.debug2.timestamps, "score": result.debug2.score}).set_index("t")
        st.line_chart(df2)
        _show_anchor_window("Clip2", result.debug2)


def run_app() -> None:
    st.set_page_config(page_title="maimai timing align", layout="wide")
    st.title("maimai-timing-align")
    st.caption("单明显转场自动对齐（极性无关）+ 完整视频导出（Clip1画面 + 双轨混音）")

    try:
        ensure_ffmpeg_available()
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        clip1 = st.file_uploader("上传 Clip1（输出画面来源）", type=["mp4", "mov", "mkv", "webm"])
    with c2:
        clip2 = st.file_uploader("上传 Clip2（对齐与混音来源）", type=["mp4", "mov", "mkv", "webm"])

    st.sidebar.header("参数")
    sample_fps = st.sidebar.slider("分析采样 FPS", 6, 30, 15)
    resize_width = st.sidebar.slider("分析宽度", 320, 1280, 640, step=32)
    center_crop_ratio = st.sidebar.slider("中心裁剪比例", 0.50, 1.00, 0.82, step=0.01)
    smoothing_window = st.sidebar.slider("平滑窗口", 3, 31, 7, step=2)
    min_peak_distance_sec = st.sidebar.slider("最小峰间隔(秒)", 0.10, 2.00, 0.35, step=0.05)
    min_peak_z = st.sidebar.slider("最小峰值z", 1.0, 8.0, 2.8, step=0.1)
    audio2_gain_db = st.sidebar.slider("Clip2 混音增益(dB)", -18.0, 6.0, -6.0, step=0.5)

    with st.sidebar.expander("高级：锚点权重", expanded=False):
        anchor_front_bias_strength = st.slider("前段偏置强度", 0.0, 8.0, 3.4, step=0.1)
        anchor_late_penalty_from = st.slider("晚段惩罚起点(相对位置)", 0.40, 0.95, 0.65, step=0.01)
        anchor_late_penalty = st.slider("晚段惩罚系数", 0.10, 1.00, 0.35, step=0.01)
        anchor_prev_pair_boost = st.slider("前邻峰加权", 1.00, 2.00, 1.25, step=0.01)
        anchor_next_pair_penalty = st.slider("后邻峰惩罚", 0.50, 1.00, 0.92, step=0.01)
        clip2_anchor_prev_pair_boost = st.slider("Clip2 前邻峰加权", 1.00, 3.00, 1.65, step=0.01)
        clip2_anchor_first_in_pair_penalty = st.slider("Clip2 首峰惩罚", 0.20, 1.00, 0.62, step=0.01)

    with st.sidebar.expander("高级：整体匹配", expanded=False):
        global_search_range_sec = st.slider("全局搜索范围(±秒)", 5.0, 60.0, 20.0, step=0.5)
        global_scan_step_sec = st.slider("全局扫描步长(秒)", 0.005, 0.100, 0.020, step=0.005)
        global_match_min_overlap_sec = st.slider("最小重叠时长(秒)", 8.0, 60.0, 20.0, step=1.0)
        global_match_window_sec = st.slider("分段窗口时长(秒)", 6.0, 40.0, 18.0, step=1.0)
        global_low_conf_global_weight = st.slider("低置信全局混合权重", 0.00, 1.00, 0.25, step=0.01)
        global_confidence_floor = st.slider("全局置信阈值", 0.00, 1.00, 0.45, step=0.01)
        global_refine_radius_sec = st.slider("精修半径(秒)", 0.10, 3.00, 1.00, step=0.05)

    use_manual = st.sidebar.checkbox("手动锚点覆盖")
    manual_anchor1 = None
    manual_anchor2 = None
    if use_manual:
        manual_anchor1 = st.sidebar.number_input("Clip1 锚点(秒)", min_value=0.0, value=0.0, step=0.1)
        manual_anchor2 = st.sidebar.number_input("Clip2 锚点(秒)", min_value=0.0, value=0.0, step=0.1)

    config = AlignConfig(
        sample_fps=float(sample_fps),
        resize_width=int(resize_width),
        center_crop_ratio=float(center_crop_ratio),
        smoothing_window=int(smoothing_window),
        min_peak_distance_sec=float(min_peak_distance_sec),
        min_peak_z=float(min_peak_z),
        anchor_front_bias_strength=float(anchor_front_bias_strength),
        anchor_late_penalty_from=float(anchor_late_penalty_from),
        anchor_late_penalty=float(anchor_late_penalty),
        anchor_prev_pair_boost=float(anchor_prev_pair_boost),
        anchor_next_pair_penalty=float(anchor_next_pair_penalty),
        clip2_anchor_prev_pair_boost=float(clip2_anchor_prev_pair_boost),
        clip2_anchor_first_in_pair_penalty=float(clip2_anchor_first_in_pair_penalty),
        global_search_range_sec=float(global_search_range_sec),
        global_scan_step_sec=float(global_scan_step_sec),
        global_match_min_overlap_sec=float(global_match_min_overlap_sec),
        global_match_window_sec=float(global_match_window_sec),
        global_low_conf_global_weight=float(global_low_conf_global_weight),
        global_confidence_floor=float(global_confidence_floor),
        global_refine_radius_sec=float(global_refine_radius_sec),
        audio2_gain_db=float(audio2_gain_db),
    )

    run = st.button("分析转场并导出完整视频", type="primary", disabled=not (clip1 and clip2))

    if run and clip1 and clip2:
        with st.spinner("处理中，请稍候..."):
            try:
                with tempfile.TemporaryDirectory(prefix="maimai-align-") as td:
                    tmp = Path(td)
                    p1 = _save_uploaded(clip1, tmp)
                    p2 = _save_uploaded(clip2, tmp)

                    result = align_videos(
                        p1,
                        p2,
                        config,
                        manual_anchor1=manual_anchor1 if use_manual else None,
                        manual_anchor2=manual_anchor2 if use_manual else None,
                    )

                    out_name = f"aligned_{Path(clip1.name).stem}_{Path(clip2.name).stem}.mp4"
                    out_path = tmp / out_name
                    export_aligned_video(p1, p2, out_path, result, config)

                    data = out_path.read_bytes()
                    result.output_path = out_path
                    st.session_state["last_output_name"] = out_name
                    st.session_state["last_output_data"] = data
                    st.session_state["last_result"] = result
                    st.session_state["last_debug_frames"] = {
                        "Clip1": _collect_debug_frames(p1, result.debug1, max_around=20),
                        "Clip2": _collect_debug_frames(p2, result.debug2, max_around=20),
                    }

            except Exception as exc:  # noqa: BLE001
                st.exception(exc)

    result = st.session_state.get("last_result")
    if result:
        st.subheader("对齐结果")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clip1 锚点", _fmt_sec(result.clip1_anchor_sec))
        m2.metric("Clip2 锚点", _fmt_sec(result.clip2_anchor_sec))
        m3.metric("偏移 offset", _fmt_sec(result.offset_sec))
        m4.metric("可导出时长", _fmt_sec(result.output_duration_sec))
        st.progress(min(1.0, max(0.0, result.confidence)))
        st.caption(f"置信度：{result.confidence:.3f}")

        _show_debug(result, st.session_state.get("last_debug_frames"))

    output_data = st.session_state.get("last_output_data")
    output_name = st.session_state.get("last_output_name", "aligned.mp4")
    if output_data:
        st.subheader("下载")
        st.download_button(
            "下载导出视频",
            data=output_data,
            file_name=output_name,
            mime="video/mp4",
        )


if __name__ == "__main__":
    run_app()
