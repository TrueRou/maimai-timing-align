from __future__ import annotations

import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZipFile

import streamlit as st

try:
    from .exporter import (
        export_aligned_video,
        export_full_clip_overlay_video,
        export_multi_segment_videos,
        export_preview_video,
        export_segment_to_end_video,
        export_segment_video,
    )
    from .media import ensure_ffmpeg_available
    from .models import AlignConfig
except ImportError:  # pragma: no cover
    from exporter import (
        export_aligned_video,
        export_full_clip_overlay_video,
        export_multi_segment_videos,
        export_preview_video,
        export_segment_to_end_video,
        export_segment_video,
    )
    from media import ensure_ffmpeg_available
    from models import AlignConfig

from maimai_timing_align.analysis import align_audio_media, batch_match_osu_candidates

MAX_UPLOAD_BYTES = 500 * 1024 * 1024
CHUNK_SIZE = 2 * 1024 * 1024
LXNS_BASE = "https://assets2.lxns.net/maimai"

def _fmt_sec(v: float) -> str:
    return f"{v:.3f}s"


def _fmt_range(start: float, duration: float) -> str:
    end = start + duration
    return f"{start:.3f}s → {end:.3f}s"


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

    st.sidebar.header("对齐引擎")
    align_mode = st.sidebar.radio(
        "对齐模式",
        options=["standard", "similar_segments"],
        format_func=lambda x: "普通音频对齐" if x == "standard" else "相似段匹配对齐",
        index=0,
    )
    align_backend = st.sidebar.radio(
        "对齐方式",
        options=["remote", "local"],
        format_func=lambda x: "外部 API" if x == "remote" else "内置 audalign",
        index=1 if align_mode == "similar_segments" else 0,
        disabled=align_mode == "similar_segments",
    )

    st.sidebar.header("API 设置")
    otoge_base_url = st.sidebar.text_input("otoge-service URL", value="https://api.turou.fun/otoge")
    otoge_developer_token = st.sidebar.text_input(
        "Developer Token",
        value="c4f33434df6d5e22a91283e68c9899f9",
        type="password",
        disabled=align_backend != "remote",
    )
    if align_backend == "local":
        st.sidebar.caption("当前使用内置 audalign，本次不会访问外部 API。")

    with st.sidebar.expander("高级：音频对齐参数", expanded=False):
        audio_sr = st.number_input("sample_rate", min_value=8000, max_value=96000, value=22050, step=50)
        audio_hop_length = st.number_input("hop_length", min_value=64, max_value=4096, value=512, step=64)
        audio_n_fft = st.number_input("n_fft", min_value=256, max_value=8192, value=2048, step=256)
        audio_search_range_sec = st.slider("search_range_sec", 3.0, 60.0, 20.0, step=0.5)
        audio_min_overlap_sec = st.slider("min_overlap_sec", 3.0, 60.0, 15.0, step=0.5)
        audio_confidence_floor = st.slider("confidence_floor", 0.00, 1.00, 0.35, step=0.01)
        audio_max_duration_sec = st.slider("max_duration_sec", 30.0, 900.0, 300.0, step=10.0)
        otoge_timeout_sec = st.slider("request_timeout_sec", 5.0, 90.0, 30.0, step=1.0)

    with st.sidebar.expander("高级：相似段匹配参数", expanded=align_mode == "similar_segments"):
        similar_match_window_sec = st.slider("seed_window_sec", 4.0, 240.0, 12.0, step=0.5)
        similar_match_step_sec = st.slider("match_step_sec", 0.5, 8.0, 2.0, step=0.5)
        similar_similarity_floor = st.slider("similarity_floor", 0.00, 1.00, 0.58, step=0.01)
        similar_max_segments = st.slider("max_segments", 1, 12, 6, step=1)
        similar_min_segment_gap_sec = st.slider("min_segment_gap_sec", 0.0, 15.0, 5.0, step=0.5)
        similar_margin_before_sec = st.slider("margin_before_sec", 0.0, 5.0, 1.5, step=0.1)
        similar_margin_after_sec = st.slider("margin_after_sec", 0.0, 5.0, 2.0, step=0.1)
        similar_export_mode = st.radio(
            "similar export mode",
            options=["segment_exports", "full_clip_overlay"],
            format_func=lambda x: "导出多个命中段落" if x == "segment_exports" else "导出 Clip1 整段并叠加命中段",
            index=0,
        )
        st.caption("`seed_window_sec` 仅作为初始搜索窗口，实际命中段长度会在分析过程中自动扩展和修正。")
        if similar_export_mode == "full_clip_overlay":
            st.caption("整段叠加模式会仅保留时间轴正序且不重复的段落，并将它们按命中位置叠加到 Clip1 全片音频上。")

    st.subheader("输入素材")
    left_col, right_col = st.columns(2)

    with left_col:
        clip1 = st.file_uploader("Clip1（手元视频）", type=["mp4", "mov", "mkv", "webm"])

    clip2_uploaded = None
    song_id = ""
    clip2_source = "upload"
    osu_bpm = 0.0
    osu_query = ""
    osu_artist = ""
    osu_creator = ""
    osu_version = ""
    osu_client_id = ""
    osu_client_secret = ""
    osu_mode = "any"
    osu_category = "has_leaderboard"
    osu_batch_limit = 5
    with right_col:
        clip2_source = st.radio(
            "Clip2 来源",
            options=["upload", "lxns_song", "osu_batch"],
            format_func=lambda x: {
                "upload": "上传 Clip2 文件",
                "lxns_song": "使用 Song ID 下载",
                "osu_batch": "从 osu 按 BPM 批量找歌",
            }[x],
            index=0,
        )
        clip2_uploaded = st.file_uploader("Clip2（音频来源）", type=["mp4", "mov", "mkv", "webm", "mp3", "wav", "m4a", "aac", "flac", "ogg", "opus"])
        if clip2_source == "lxns_song":
            song_id = st.text_input("Song ID（仅数字）", value="").strip()
        elif clip2_source == "osu_batch":
            osu_bpm = st.number_input("osu BPM", min_value=1.0, max_value=400.0, value=180.0, step=1.0)
            osu_query = st.text_input("关键词", value="").strip()
            osu_artist = st.text_input("艺术家筛选", value="").strip()
            osu_creator = st.text_input("谱师筛选", value="").strip()
            osu_version = st.text_input("难度名筛选", value="").strip()
            osu_mode = st.selectbox("模式", options=["any", "osu", "taiko", "fruits", "mania"], index=0)
            osu_category = st.selectbox(
                "分类",
                options=["has_leaderboard", "ranked", "loved", "qualified", "pending", "graveyard"],
                index=0,
            )
            osu_batch_limit = st.slider("候选数量", 1, 20, 5, step=1)
            osu_client_id = st.text_input("osu Client ID", value="")
            osu_client_secret = st.text_input("osu Client Secret", value="", type="password")

    config = AlignConfig(
        otoge_base_url=str(otoge_base_url).strip(),
        otoge_developer_token=str(otoge_developer_token).strip(),
        otoge_timeout_sec=float(otoge_timeout_sec),
        align_mode=str(align_mode), # type: ignore
        align_backend=str(align_backend), # type: ignore
        align_fallback_to_local=True,
        audio_sr=int(audio_sr),
        audio_hop_length=int(audio_hop_length),
        audio_n_fft=int(audio_n_fft),
        audio_search_range_sec=float(audio_search_range_sec),
        audio_min_overlap_sec=float(audio_min_overlap_sec),
        audio_confidence_floor=float(audio_confidence_floor),
        audio_max_duration_sec=float(audio_max_duration_sec),
        similar_match_window_sec=float(similar_match_window_sec),
        similar_match_step_sec=float(similar_match_step_sec),
        similar_similarity_floor=float(similar_similarity_floor),
        similar_max_segments=int(similar_max_segments),
        similar_min_segment_gap_sec=float(similar_min_segment_gap_sec),
        similar_margin_before_sec=float(similar_margin_before_sec),
        similar_margin_after_sec=float(similar_margin_after_sec),
        similar_export_mode=str(similar_export_mode), # type: ignore
        osu_client_id=str(osu_client_id).strip(),
        osu_client_secret=str(osu_client_secret).strip(),
        osu_query=str(osu_query).strip(),
        osu_artist=str(osu_artist).strip(),
        osu_creator=str(osu_creator).strip(),
        osu_version=str(osu_version).strip(),
        osu_bpm=float(osu_bpm),
        osu_mode=str(osu_mode), # type: ignore
        osu_category=str(osu_category), # type: ignore
        osu_batch_limit=int(osu_batch_limit),
        audio1_gain_db=float(audio1_gain_db),
        audio2_gain_db=float(audio2_gain_db),
        audio_reverb_wet=float(audio_reverb_wet),
        output_video_codec=str(output_video_codec),
        output_target_preset=str(output_target_preset),
        output_width=int(output_width),
        output_fps=int(output_fps),
    )

    run_align = st.button("开始对齐", type="primary", disabled=not bool(clip1))

    if run_align and clip1:
        with st.spinner("处理中，请稍候..."):
            try:
                result = None
                p1 = _save_uploaded(
                    clip1,
                    in_dir / f"clip1_{_safe_name(Path(clip1.name).stem, 'clip1')}{Path(clip1.name).suffix}",
                )

                if clip2_source == "upload" and song_id and clip2_uploaded:
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
                elif clip2_source == "lxns_song" and not song_id and not clip2_uploaded:
                    raise RuntimeError("请提供 Clip2 的视频/音频文件或有效的 Song ID")
                elif clip2_source == "lxns_song" and song_id and not clip2_uploaded:
                    p2 = _download_lxns_song(song_id, in_dir / f"clip2_song_{song_id}.mp3")
                    clip2_name = f"song_{song_id}.mp3"
                elif clip2_source == "upload" and clip2_uploaded:
                    p2 = _save_uploaded(
                        clip2_uploaded,
                        in_dir
                        / (
                            f"clip2_{_safe_name(Path(clip2_uploaded.name).stem, 'clip2')}"
                            f"{Path(clip2_uploaded.name).suffix}"
                        ),
                    )
                    clip2_name = clip2_uploaded.name
                elif clip2_source == "osu_batch":
                    batch_results = batch_match_osu_candidates(p1, in_dir / "osu_batch", config)
                    best = batch_results[0]
                    p2 = best.audio_path
                    clip2_name = f"{best.candidate.artist} - {best.candidate.title} [{best.candidate.version}]"
                    st.session_state["osu_batch_results"] = batch_results
                    result = best.result
                else:
                    raise RuntimeError("无法确定 Clip2 的输入来源，请检查上传的文件和 Song ID")

                if clip2_source != "osu_batch":
                    result = align_audio_media(p1, p2, config)

                st.session_state["clip1_path"] = str(p1)
                st.session_state["clip2_path"] = str(p2)
                st.session_state["clip1_name"] = clip1.name
                st.session_state["clip2_name"] = clip2_name
                st.session_state["last_result"] = result
                st.session_state["preview_path"] = ""
                st.session_state["final_output_path"] = ""
                st.session_state["segment_output_paths"] = []
                st.session_state["segment_zip_path"] = ""
                if clip2_source != "osu_batch":
                    st.session_state["osu_batch_results"] = []

            except Exception as exc:  # noqa: BLE001
                st.exception(exc)

    result = st.session_state.get("last_result")
    if result:
        batch_results = st.session_state.get("osu_batch_results")
        if batch_results:
            st.subheader("osu 批量匹配结果")
            for idx, item in enumerate(batch_results, start=1):
                with st.container(border=True):
                    badge = "⭐ 最佳歌曲" if idx == 1 else f"候选 #{idx}"
                    st.markdown(f"**{badge}**")
                    st.caption(
                        f"{item.candidate.artist} - {item.candidate.title} [{item.candidate.version}] | "
                        f"Mapper: {item.candidate.creator} | BPM: {item.candidate.bpm:.0f} | 分数: {item.result.confidence:.3f}"
                    )
                    if item.result.segments:
                        with st.expander(f"查看 {badge} 的候选段落", expanded=idx == 1):
                            for segment in item.result.segments:
                                st.markdown(
                                    f"- {'⭐ ' if segment.is_best else ''}段落 #{segment.rank} | "
                                    f"Clip1 {_fmt_range(segment.clip1_match_start_sec, segment.match_duration_sec)} | "
                                    f"Clip2 {_fmt_range(segment.clip2_match_start_sec, segment.match_duration_sec)} | "
                                    f"综合分数 {segment.score:.3f}"
                                )
                                st.caption(
                                    f"导出范围：Clip1 {_fmt_range(segment.clip1_export_start_sec, segment.export_duration_sec)} | "
                                    f"Clip2 {_fmt_range(segment.clip2_export_start_sec, segment.export_duration_sec)}"
                                )
                                preview_now = st.button(
                                    f"预览 {badge} 段落 #{segment.rank}",
                                    key=f"preview_batch_{idx}_{segment.segment_id}",
                                )
                                if preview_now:
                                    with st.spinner(f"正在生成 {badge} 段落 #{segment.rank} 预览..."):
                                        try:
                                            p1 = Path(st.session_state["clip1_path"])
                                            p2 = item.audio_path
                                            preview_name = (
                                                f"preview_{idx:02d}_{_safe_name(item.candidate.artist, 'artist')}_"
                                                f"{_safe_name(item.candidate.title, 'title')}_{segment.rank:02d}.mp4"
                                            )
                                            preview_path = out_dir / preview_name
                                            export_preview_video(
                                                clip1_path=p1,
                                                clip2_path=p2,
                                                output_path=preview_path,
                                                result=type(item.result)(
                                                    clip1_anchor_sec=item.result.clip1_anchor_sec,
                                                    clip2_anchor_sec=item.result.clip2_anchor_sec,
                                                    offset_sec=item.result.offset_sec,
                                                    clip1_start_sec=segment.clip1_export_start_sec,
                                                    clip2_start_sec=segment.clip2_export_start_sec,
                                                    output_duration_sec=segment.export_duration_sec,
                                                    confidence=segment.score,
                                                    method=item.result.method,
                                                    warnings=item.result.warnings,
                                                    segments=[segment],
                                                    best_segment_index=0,
                                                ),
                                                config=config,
                                                mute_clip1=False,
                                                mute_clip2=False,
                                            )
                                            with preview_path.open("rb") as fr:
                                                st.download_button(
                                                    f"下载预览 {badge} 段落 #{segment.rank}",
                                                    data=fr,
                                                    file_name=preview_path.name,
                                                    mime="video/mp4",
                                                    key=f"download_preview_batch_{idx}_{segment.segment_id}",
                                                )
                                        except Exception as exc:  # noqa: BLE001
                                            st.exception(exc)

                                export_segment_batch_now = st.button(
                                    f"导出 {badge} 段落 #{segment.rank}",
                                    key=f"export_batch_segment_{idx}_{segment.segment_id}",
                                )
                                if export_segment_batch_now:
                                    with st.spinner(f"正在导出 {badge} 段落 #{segment.rank}..."):
                                        try:
                                            p1 = Path(st.session_state["clip1_path"])
                                            p2 = item.audio_path
                                            out_name = (
                                                f"aligned_{idx:02d}_{_safe_name(item.candidate.artist, 'artist')}_"
                                                f"{_safe_name(item.candidate.title, 'title')}_{segment.rank:02d}.mp4"
                                            )
                                            out_path = out_dir / out_name
                                            export_segment_video(
                                                clip1_path=p1,
                                                clip2_path=p2,
                                                output_path=out_path,
                                                segment=segment,
                                                config=config,
                                                mute_clip1=False,
                                                mute_clip2=False,
                                            )
                                            with out_path.open("rb") as fr:
                                                st.download_button(
                                                    f"下载 {badge} 段落 #{segment.rank}",
                                                    data=fr,
                                                    file_name=out_path.name,
                                                    mime="video/mp4",
                                                    key=f"download_batch_segment_{idx}_{segment.segment_id}",
                                                )
                                        except Exception as exc:  # noqa: BLE001
                                            st.exception(exc)

                                export_to_end_batch_now = st.button(
                                    f"从 {badge} 段落 #{segment.rank} 导出到结尾",
                                    key=f"export_batch_to_end_{idx}_{segment.segment_id}",
                                )
                                if export_to_end_batch_now:
                                    with st.spinner(f"正在导出 {badge} 段落 #{segment.rank} 到结尾..."):
                                        try:
                                            p1 = Path(st.session_state["clip1_path"])
                                            p2 = item.audio_path
                                            out_name = (
                                                f"aligned_{idx:02d}_{_safe_name(item.candidate.artist, 'artist')}_"
                                                f"{_safe_name(item.candidate.title, 'title')}_{segment.rank:02d}_to_end.mp4"
                                            )
                                            out_path = out_dir / out_name
                                            export_segment_to_end_video(
                                                clip1_path=p1,
                                                clip2_path=p2,
                                                output_path=out_path,
                                                segment=segment,
                                                config=config,
                                                mute_clip1=False,
                                                mute_clip2=False,
                                            )
                                            with out_path.open("rb") as fr:
                                                st.download_button(
                                                    f"下载 {badge} 段落 #{segment.rank} 到结尾",
                                                    data=fr,
                                                    file_name=out_path.name,
                                                    mime="video/mp4",
                                                    key=f"download_batch_to_end_{idx}_{segment.segment_id}",
                                                )
                                        except Exception as exc:  # noqa: BLE001
                                            st.exception(exc)

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

        if result.segments:
            best_segment = result.best_segment
            if best_segment is not None:
                st.info(
                    "最佳段落："
                    f"#{best_segment.rank} | Clip1 {_fmt_range(best_segment.clip1_match_start_sec, best_segment.match_duration_sec)} | "
                    f"Clip2 {_fmt_range(best_segment.clip2_match_start_sec, best_segment.match_duration_sec)}"
                )

            st.markdown("### 命中段落")
            for segment in result.segments:
                with st.container(border=True):
                    tag = "⭐ 最佳段落" if segment.is_best else f"候选段 #{segment.rank}"
                    st.markdown(f"**{tag}**")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Clip1 命中区间", _fmt_range(segment.clip1_match_start_sec, segment.match_duration_sec))
                    c2.metric("Clip2 命中区间", _fmt_range(segment.clip2_match_start_sec, segment.match_duration_sec))
                    c3.metric("导出时长", _fmt_sec(segment.export_duration_sec))
                    st.caption(
                        f"综合分数 {segment.score:.3f} | onset {segment.onset_score:.3f} | "
                        f"chroma {segment.chroma_score:.3f} | tempogram {segment.tempogram_score:.3f}"
                    )
                    st.caption(
                        "导出范围："
                        f"Clip1 {_fmt_range(segment.clip1_export_start_sec, segment.export_duration_sec)} | "
                        f"Clip2 {_fmt_range(segment.clip2_export_start_sec, segment.export_duration_sec)}"
                    )

                    export_segment_now = st.button(
                        f"导出该段 #{segment.rank}",
                        key=f"export_segment_{segment.segment_id}",
                    )
                    if export_segment_now:
                        with st.spinner(f"正在导出段落 #{segment.rank}..."):
                            try:
                                p1 = Path(st.session_state["clip1_path"])
                                p2 = Path(st.session_state["clip2_path"])
                                base_name = (
                                    f"aligned_{_safe_name(Path(st.session_state['clip1_name']).stem, 'clip1')}_"
                                    f"{_safe_name(Path(st.session_state['clip2_name']).stem, 'clip2')}"
                                )
                                out_path = out_dir / f"{base_name}_segment_{segment.rank:02d}{'_best' if segment.is_best else ''}.mp4"
                                export_segment_video(
                                    clip1_path=p1,
                                    clip2_path=p2,
                                    output_path=out_path,
                                    segment=segment,
                                    config=config,
                                    mute_clip1=False,
                                    mute_clip2=False,
                                )
                                st.success(f"段落 #{segment.rank} 导出完成")
                                with out_path.open("rb") as fr:
                                    st.download_button(
                                        f"下载段落 #{segment.rank}",
                                        data=fr,
                                        file_name=out_path.name,
                                        mime="video/mp4",
                                        key=f"download_segment_{segment.segment_id}",
                                    )
                            except Exception as exc:  # noqa: BLE001
                                st.exception(exc)

                    export_to_end_now = st.button(
                        f"从该段导出到结尾 #{segment.rank}",
                        key=f"export_segment_to_end_{segment.segment_id}",
                    )
                    if export_to_end_now:
                        with st.spinner(f"正在导出段落 #{segment.rank} 到结尾..."):
                            try:
                                p1 = Path(st.session_state["clip1_path"])
                                p2 = Path(st.session_state["clip2_path"])
                                base_name = (
                                    f"aligned_{_safe_name(Path(st.session_state['clip1_name']).stem, 'clip1')}_"
                                    f"{_safe_name(Path(st.session_state['clip2_name']).stem, 'clip2')}"
                                )
                                out_path = out_dir / (
                                    f"{base_name}_segment_{segment.rank:02d}_to_end"
                                    f"{'_best' if segment.is_best else ''}.mp4"
                                )
                                export_segment_to_end_video(
                                    clip1_path=p1,
                                    clip2_path=p2,
                                    output_path=out_path,
                                    segment=segment,
                                    config=config,
                                    mute_clip1=False,
                                    mute_clip2=False,
                                )
                                st.success(f"段落 #{segment.rank} 到结尾导出完成")
                                with out_path.open("rb") as fr:
                                    st.download_button(
                                        f"下载段落 #{segment.rank} 到结尾",
                                        data=fr,
                                        file_name=out_path.name,
                                        mime="video/mp4",
                                        key=f"download_segment_to_end_{segment.segment_id}",
                                    )
                            except Exception as exc:  # noqa: BLE001
                                st.exception(exc)

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
                    if result.segments and config.align_mode == "similar_segments" and config.similar_export_mode == "full_clip_overlay":
                        export_full_clip_overlay_video(
                            clip1_path=p1,
                            clip2_path=p2,
                            output_path=out_path,
                            result=result,
                            config=config,
                        )
                    else:
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

        if result.segments and config.similar_export_mode == "segment_exports":
            export_all_segments = st.button("批量导出所有命中段落")
            if export_all_segments:
                with st.spinner("正在批量导出段落..."):
                    try:
                        p1 = Path(st.session_state["clip1_path"])
                        p2 = Path(st.session_state["clip2_path"])
                        base_name = (
                            f"aligned_{_safe_name(Path(st.session_state['clip1_name']).stem, 'clip1')}_"
                            f"{_safe_name(Path(st.session_state['clip2_name']).stem, 'clip2')}"
                        )
                        exported = export_multi_segment_videos(
                            clip1_path=p1,
                            clip2_path=p2,
                            output_dir=out_dir,
                            result=result,
                            config=config,
                            base_name=base_name,
                            mute_clip1=False,
                            mute_clip2=False,
                        )
                        zip_path = out_dir / f"{base_name}_segments.zip"
                        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zf:
                            for path in exported:
                                zf.write(path, arcname=path.name)
                        st.session_state["segment_output_paths"] = [str(path) for path in exported]
                        st.session_state["segment_zip_path"] = str(zip_path)
                        st.success(f"已导出 {len(exported)} 个段落")
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

        segment_zip_path = st.session_state.get("segment_zip_path")
        if segment_zip_path and Path(segment_zip_path).exists():
            with Path(segment_zip_path).open("rb") as fr:
                st.download_button(
                    "下载全部命中段落(zip)",
                    data=fr,
                    file_name=Path(segment_zip_path).name,
                    mime="application/zip",
                )


if __name__ == "__main__":
    run_app()
