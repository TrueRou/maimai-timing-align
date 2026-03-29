from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

import streamlit as st
from exporter import export_aligned_video
from media import ensure_ffmpeg_available
from models import AlignConfig

from maimai_timing_align.analysis import align_audio_media

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
CHUNK_SIZE = 2 * 1024 * 1024
LXNS_BASE = "https://assets2.lxns.net/maimai"

def _fmt_sec(v: float) -> str:
    return f"{v:.3f}s"


def _safe_name(name: str, fallback: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name.strip())
    return (out[:80] or fallback).strip("_")


def _save_uploaded(uploaded, target_path: Path) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    uploaded.seek(0)
    with target_path.open("wb") as fw:
        while True:
            chunk = uploaded.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise RuntimeError("文件超过 500MB 限制")
            fw.write(chunk)
    uploaded.seek(0)
    return target_path


def _download_lxns_song(song_id: str, target_path: Path) -> Path:
    sid = (song_id or "").strip()
    if not sid.isdigit():
        raise RuntimeError("谱面 Song ID 必须是数字")
    req = Request(
        f"{LXNS_BASE}/music/{sid}.mp3",
        headers={"User-Agent": "maimai-timing-align/0.2", "Accept": "audio/mpeg,*/*;q=0.8"},
    )
    total = 0
    with urlopen(req, timeout=30) as resp, target_path.open("wb") as fw:  # noqa: S310
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise RuntimeError("下载音频超过 500MB 限制")
            fw.write(chunk)
    return target_path


def run_app() -> None:
    st.set_page_config(page_title="maimai timing align", layout="wide")
    st.title("maimai Timing Align")
    st.caption("基于音频内容对齐两段视频/音频，为你的手元呈现完整的听觉体验！")

    if "work_dir" not in st.session_state:
        st.session_state["work_dir"] = tempfile.mkdtemp(prefix="maimai-align-ui-")
    work_dir = Path(st.session_state["work_dir"])
    in_dir = work_dir / "inputs"
    out_dir = work_dir / "outputs"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ensure_ffmpeg_available()
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

    st.sidebar.header("导出设置")
    output_target_preset = st.sidebar.selectbox(
        "导出预设",
        options=["aggressive", "small", "balanced", "quality"],
        format_func=lambda x: {
            "aggressive": "极限压缩",
            "small": "体积优先",
            "balanced": "平衡",
            "quality": "画质优先",
        }[x],
        index=1,
    )
    output_video_codec = st.sidebar.selectbox(
        "编码格式",
        options=["h264", "h265"],
        format_func=lambda x: "H.264 (兼容优先)" if x == "h264" else "H.265 (体积更小)",
        index=1,
    )
    output_width = st.sidebar.selectbox("输出宽度", options=[720, 1080, 1440], index=1)
    output_fps = st.sidebar.selectbox("输出帧率", options=[30, 60], index=0)

    st.sidebar.header("音频混合")
    audio1_gain_db = st.sidebar.slider("A 轨(Clip1)响度(dB)", -18.0, 6.0, 0.0, step=0.5)
    audio2_gain_db = st.sidebar.slider("B 轨(Clip2)响度(dB)", -18.0, 6.0, -6.0, step=0.5)
    audio_reverb_wet = st.sidebar.slider("混响强度", 0.0, 0.6, 0.12, step=0.01)

    st.sidebar.header("API 设置")
    otoge_base_url = st.sidebar.text_input("otoge-service URL", value="https://api.turou.fun/otoge")
    otoge_developer_token = st.sidebar.text_input("Developer Token", value="c4f33434df6d5e22a91283e68c9899f9", type="password")

    with st.sidebar.expander("高级：音频对齐参数", expanded=False):
        audio_sr = st.number_input("sample_rate", min_value=8000, max_value=96000, value=22050, step=50)
        audio_hop_length = st.number_input("hop_length", min_value=64, max_value=4096, value=512, step=64)
        audio_n_fft = st.number_input("n_fft", min_value=256, max_value=8192, value=2048, step=256)
        audio_search_range_sec = st.slider("search_range_sec", 3.0, 60.0, 20.0, step=0.5)
        audio_min_overlap_sec = st.slider("min_overlap_sec", 3.0, 60.0, 15.0, step=0.5)
        audio_confidence_floor = st.slider("confidence_floor", 0.00, 1.00, 0.35, step=0.01)
        audio_max_duration_sec = st.slider("max_duration_sec", 30.0, 900.0, 300.0, step=10.0)
        otoge_timeout_sec = st.slider("request_timeout_sec", 5.0, 90.0, 30.0, step=1.0)

    config = AlignConfig(
        otoge_base_url=str(otoge_base_url).strip(),
        otoge_developer_token=str(otoge_developer_token).strip(),
        otoge_timeout_sec=float(otoge_timeout_sec),
        audio_sr=int(audio_sr),
        audio_hop_length=int(audio_hop_length),
        audio_n_fft=int(audio_n_fft),
        audio_search_range_sec=float(audio_search_range_sec),
        audio_min_overlap_sec=float(audio_min_overlap_sec),
        audio_confidence_floor=float(audio_confidence_floor),
        audio_max_duration_sec=float(audio_max_duration_sec),
        audio1_gain_db=float(audio1_gain_db),
        audio2_gain_db=float(audio2_gain_db),
        audio_reverb_wet=float(audio_reverb_wet),
        output_video_codec=str(output_video_codec),
        output_target_preset=str(output_target_preset),
        output_width=int(output_width),
        output_fps=int(output_fps),
    )

    st.subheader("输入素材")
    left_col, right_col = st.columns(2)

    with left_col:
        clip1 = st.file_uploader("Clip1（手元视频）", type=["mp4", "mov", "mkv", "webm"])

    clip2_uploaded = None
    song_id = ""
    with right_col:
        clip2_uploaded = st.file_uploader("Clip2（音频来源）", type=["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a", "aac", "flac", "ogg", "opus"])
        song_id = st.text_input("Song ID（仅数字）", value="").strip()

    run_align = st.button("开始对齐", type="primary", disabled=not bool(clip1))

    if run_align and clip1:
        with st.spinner("处理中，请稍候..."):
            try:
                p1 = _save_uploaded(
                    clip1,
                    in_dir / f"clip1_{_safe_name(Path(clip1.name).stem, 'clip1')}{Path(clip1.name).suffix}",
                )

                if song_id and clip2_uploaded:
                    st.warning("同时提供了 Clip2 文件和 Song ID，将优先使用上传的文件进行对齐")
                    p2 = _save_uploaded(
                        clip2_uploaded,
                        in_dir
                        / (
                            f"clip2_{_safe_name(Path(clip2_uploaded.name).stem, 'clip2')}"
                            f"{Path(clip2_uploaded.name).suffix}"
                        ),
                    )
                    clip2_name = clip2_uploaded.name
                elif not song_id and not clip2_uploaded:
                    raise RuntimeError("请提供 Clip2 的视频/音频文件或有效的 Song ID")
                elif song_id and not clip2_uploaded:
                    p2 = _download_lxns_song(song_id, in_dir / f"clip2_song_{song_id}.mp3")
                    clip2_name = f"song_{song_id}.mp3"
                elif clip2_uploaded and not song_id:
                    p2 = _save_uploaded(
                        clip2_uploaded,
                        in_dir
                        / (
                            f"clip2_{_safe_name(Path(clip2_uploaded.name).stem, 'clip2')}"
                            f"{Path(clip2_uploaded.name).suffix}"
                        ),
                    )
                    clip2_name = clip2_uploaded.name
                else:
                    raise RuntimeError("无法确定 Clip2 的输入来源，请检查上传的文件和 Song ID")

                result = align_audio_media(p1, p2, config)

                st.session_state["clip1_path"] = str(p1)
                st.session_state["clip2_path"] = str(p2)
                st.session_state["clip1_name"] = clip1.name
                st.session_state["clip2_name"] = clip2_name
                st.session_state["last_result"] = result
                st.session_state["preview_path"] = ""
                st.session_state["final_output_path"] = ""

            except Exception as exc:  # noqa: BLE001
                st.exception(exc)

    result = st.session_state.get("last_result")
    if result:
        st.subheader("对齐结果")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Clip1 起点", _fmt_sec(result.clip1_start_sec))
        m2.metric("Clip2 起点", _fmt_sec(result.clip2_start_sec))
        m3.metric("偏移", _fmt_sec(result.offset_sec))
        m4.metric("可导出时长", _fmt_sec(result.output_duration_sec))
        st.progress(min(1.0, max(0.0, float(result.confidence))))
        st.caption(f"置信度：{result.confidence:.3f} | 方法：{result.method}")
        if result.warnings:
            st.warning("\n".join(result.warnings))

        export_now = st.button("导出完整视频", type="primary")
        if export_now:
            with st.spinner("正在导出..."):
                try:
                    p1 = Path(st.session_state["clip1_path"])
                    p2 = Path(st.session_state["clip2_path"])
                    out_name = (
                        f"aligned_{_safe_name(Path(st.session_state['clip1_name']).stem, 'clip1')}_"
                        f"{_safe_name(Path(st.session_state['clip2_name']).stem, 'clip2')}.mp4"
                    )
                    out_path = out_dir / out_name
                    export_aligned_video(
                        clip1_path=p1,
                        clip2_path=p2,
                        output_path=out_path,
                        result=result,
                        config=config,
                        mute_clip1=False,
                        mute_clip2=False,
                    )
                    st.session_state["final_output_path"] = str(out_path)
                except Exception as exc:  # noqa: BLE001
                    st.exception(exc)

        final_output_path = st.session_state.get("final_output_path")
        if final_output_path and Path(final_output_path).exists():
            st.success("导出完成")
            size_mb = os.path.getsize(final_output_path) / (1024 * 1024)
            st.caption(f"文件大小：{size_mb:.2f} MB")
            with Path(final_output_path).open("rb") as fr:
                st.download_button(
                    "下载导出视频",
                    data=fr,
                    file_name=Path(final_output_path).name,
                    mime="video/mp4",
                )


if __name__ == "__main__":
    run_app()
