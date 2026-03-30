from __future__ import annotations

import math
import shlex
import subprocess
from pathlib import Path

try:
    from .media import ensure_ffmpeg_available, probe_media
    from .models import AlignConfig, AlignResult, AlignSegment
except ImportError:  # pragma: no cover
    from media import ensure_ffmpeg_available, probe_media
    from models import AlignConfig, AlignResult, AlignSegment


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def _codec_candidates(codec: str) -> list[tuple[str, list[str]]]:
    c = (codec or "h264").strip().lower()
    if c == "h265":
        return [
            ("hevc_videotoolbox", ["-b:v", "2800k"]),
            ("hevc_nvenc", ["-cq", "30"]),
            ("libx265", ["-crf", "31", "-preset", "faster"]),
        ]
    return [
        ("h264_videotoolbox", ["-b:v", "3000k"]),
        ("h264_nvenc", ["-cq", "28"]),
        ("libx264", ["-crf", "30", "-preset", "faster"]),
    ]


def _preset_crf(preset_name: str) -> tuple[int, str]:
    p = (preset_name or "balanced").lower()
    if p == "aggressive":
        return 32, "veryfast"
    if p == "small":
        return 29, "faster"
    if p == "quality":
        return 24, "slow"
    return 27, "medium"


def _audio_mix_chain(config: AlignConfig, mute_clip1: bool, mute_clip2: bool) -> str:
    clip1_volume = 0.0 if mute_clip1 else 10 ** (float(config.audio1_gain_db) / 20.0)
    clip2_volume = 0.0 if mute_clip2 else 10 ** (float(config.audio2_gain_db) / 20.0)
    wet = float(max(0.0, min(1.0, config.audio_reverb_wet)))
    dry = 1.0 - wet

    return (
        f"[0:a]atrim=start={{s1}}:duration={{d}},asetpts=PTS-STARTPTS,volume={clip1_volume:.6f}[a1];"
        f"[1:a]atrim=start={{s2}}:duration={{d}},asetpts=PTS-STARTPTS,volume={clip2_volume:.6f}[a2];"
        "[a1][a2]amix=inputs=2:normalize=0:dropout_transition=0[amix];"
        f"[amix]asplit[dry][wet];[wet]aecho=0.8:0.5:40:0.35[rev];"
        f"[dry][rev]amix=inputs=2:weights='{dry:.3f} {wet:.3f}':normalize=0,alimiter=limit=0.95[aout]"
    )


def _segment_sort_key(segment: AlignSegment) -> tuple[float, float, int]:
    return (float(segment.clip1_export_start_sec), float(segment.clip2_export_start_sec), int(segment.rank))


def filter_overlay_segments(segments: list[AlignSegment], min_gap_sec: float) -> list[AlignSegment]:
    ordered = sorted(segments, key=_segment_sort_key)
    selected: list[AlignSegment] = []
    last_clip1_end = -math.inf
    last_clip2_end = -math.inf
    gap = max(0.0, float(min_gap_sec))
    for segment in ordered:
        if segment.clip1_export_start_sec + 1e-6 < last_clip1_end + gap:
            continue
        if segment.clip2_export_start_sec + 1e-6 < last_clip2_end + gap:
            continue
        selected.append(segment)
        last_clip1_end = max(last_clip1_end, float(segment.clip1_export_end_sec))
        last_clip2_end = max(last_clip2_end, float(segment.clip2_export_end_sec))
    return selected


def _build_full_clip_overlay_filter(
    clip1_duration_sec: float,
    selected_segments: list[AlignSegment],
    config: AlignConfig,
    out_width: int,
    out_fps: int,
) -> tuple[str, list[str]]:
    video = (
        f"[0:v]trim=start=0:duration={_fmt(clip1_duration_sec)},setpts=PTS-STARTPTS,"
        f"fps={int(out_fps)},scale=w={int(out_width)}:h=-2:flags=lanczos,format=yuv420p[vout]"
    )

    clip1_volume = 10 ** (float(config.audio1_gain_db) / 20.0)
    clip2_volume = 10 ** (float(config.audio2_gain_db) / 20.0)
    wet = float(max(0.0, min(1.0, config.audio_reverb_wet)))
    dry = 1.0 - wet

    parts = [video, f"[0:a]atrim=start=0:duration={_fmt(clip1_duration_sec)},asetpts=PTS-STARTPTS,volume={clip1_volume:.6f}[base0]"]
    current_label = "base0"
    for idx, segment in enumerate(selected_segments, start=1):
        overlay_label = f"ov{idx}"
        mixed_label = f"mix{idx}"
        delay_ms = max(0, int(round(float(segment.clip1_export_start_sec) * 1000.0)))
        parts.append(
            f"[1:a]atrim=start={_fmt(segment.clip2_export_start_sec)}:duration={_fmt(segment.export_duration_sec)},"
            f"asetpts=PTS-STARTPTS,volume={clip2_volume:.6f},adelay={delay_ms}|{delay_ms}[{overlay_label}]"
        )
        parts.append(
            f"[{current_label}][{overlay_label}]amix=inputs=2:normalize=0:dropout_transition=0[{mixed_label}]"
        )
        current_label = mixed_label

    parts.append(
        f"[{current_label}]asplit[dry][wet];[wet]aecho=0.8:0.5:40:0.35[rev];"
        f"[dry][rev]amix=inputs=2:weights='{dry:.3f} {wet:.3f}':normalize=0,alimiter=limit=0.95[aout]"
    )
    return ";".join(parts), ["-map", "[vout]", "-map", "[aout]"]


def _build_filter(
    result: AlignResult,
    out_duration_sec: float,
    out_width: int,
    out_fps: int,
    config: AlignConfig,
    mute_clip1: bool,
    mute_clip2: bool,
) -> tuple[str, list[str]]:
    s1 = _fmt(result.clip1_start_sec)
    s2 = _fmt(result.clip2_start_sec)
    d = _fmt(out_duration_sec)
    v = (
        f"[0:v]trim=start={s1}:duration={d},setpts=PTS-STARTPTS,"
        f"fps={int(out_fps)},scale=w={int(out_width)}:h=-2:flags=lanczos,format=yuv420p[vout]"
    )

    audio_mix = _audio_mix_chain(config, mute_clip1=mute_clip1, mute_clip2=mute_clip2)
    audio_mix = audio_mix.replace("{s1}", s1).replace("{s2}", s2).replace("{d}", d)

    fc = f"{v};{audio_mix}"
    maps = ["-map", "[vout]", "-map", "[aout]"]
    return fc, maps


def _run_ffmpeg_with_fallback(cmd_builder) -> None:
    bins = ensure_ffmpeg_available()
    last_error = ""
    for codec, extra in cmd_builder["candidates"]:
        cmd = cmd_builder["base"](bins.ffmpeg) + ["-c:v", codec, *extra] + cmd_builder["tail"]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0:
            return
        last_error = proc.stderr[-3000:]

    safe_cmd = " ".join(shlex.quote(x) for x in cmd_builder["base"](bins.ffmpeg) + cmd_builder["tail"])
    raise RuntimeError(f"导出失败。cmd: {safe_cmd}\nstderr: {last_error}")


def export_aligned_video(
    clip1_path: Path,
    clip2_path: Path,
    output_path: Path,
    result: AlignResult,
    config: AlignConfig,
    *,
    mute_clip1: bool = False,
    mute_clip2: bool = False,
    preview_duration_sec: float | None = None,
) -> Path:
    m1 = probe_media(clip1_path)
    m2 = probe_media(clip2_path)

    if not m1.has_video:
        raise RuntimeError("Clip1 必须是视频文件")
    if not m1.has_audio and not m2.has_audio:
        raise RuntimeError("至少需要一条音轨用于导出")

    crf, preset = _preset_crf(config.output_target_preset)
    out_duration = float(result.output_duration_sec)
    if preview_duration_sec is not None:
        out_duration = min(out_duration, max(2.0, float(preview_duration_sec)))

    filter_complex, map_args = _build_filter(
        result=result,
        out_duration_sec=out_duration,
        out_width=int(config.output_width),
        out_fps=int(config.output_fps),
        config=config,
        mute_clip1=mute_clip1,
        mute_clip2=mute_clip2,
    )

    def _base(ffmpeg_bin: str) -> list[str]:
        return [
            ffmpeg_bin,
            "-y",
            "-i",
            str(clip1_path),
            "-i",
            str(clip2_path),
            "-filter_complex",
            filter_complex,
            *map_args,
        ]

    tail = [
        "-c:a",
        "aac",
        "-b:a",
        f"{int(config.output_audio_bitrate_k)}k",
        "-movflags",
        "+faststart",
        "-crf",
        str(int(config.output_crf or crf)),
        "-preset",
        str(config.output_preset or preset),
        str(output_path),
    ]

    _run_ffmpeg_with_fallback(
        {
            "base": _base,
            "tail": tail,
            "candidates": _codec_candidates(config.output_video_codec),
        }
    )

    return output_path


def export_preview_video(
    clip1_path: Path,
    clip2_path: Path,
    output_path: Path,
    result: AlignResult,
    config: AlignConfig,
    *,
    mute_clip1: bool = False,
    mute_clip2: bool = False,
) -> Path:
    return export_aligned_video(
        clip1_path=clip1_path,
        clip2_path=clip2_path,
        output_path=output_path,
        result=result,
        config=config,
        mute_clip1=mute_clip1,
        mute_clip2=mute_clip2,
        preview_duration_sec=config.preview_duration_sec,
    )


def result_from_segment(segment: AlignSegment, method: str = "audio_similar_segments_local") -> AlignResult:
    return AlignResult(
        clip1_anchor_sec=float(segment.clip1_match_start_sec),
        clip2_anchor_sec=float(segment.clip2_match_start_sec),
        offset_sec=float(segment.clip2_match_start_sec - segment.clip1_match_start_sec),
        clip1_start_sec=float(segment.clip1_export_start_sec),
        clip2_start_sec=float(segment.clip2_export_start_sec),
        output_duration_sec=float(segment.export_duration_sec),
        confidence=float(segment.score),
        method=method,
        segments=[segment],
        best_segment_index=0,
    )


def result_from_segment_to_end(
    segment: AlignSegment,
    clip1_duration_sec: float,
    clip2_duration_sec: float,
    method: str = "audio_similar_segments_local_to_end",
) -> AlignResult:
    output_duration = min(
        max(0.0, float(clip1_duration_sec) - float(segment.clip1_export_start_sec)),
        max(0.0, float(clip2_duration_sec) - float(segment.clip2_export_start_sec)),
    )
    if output_duration <= 0.5:
        raise RuntimeError("该段落到结尾的可导出时长不足")

    return AlignResult(
        clip1_anchor_sec=float(segment.clip1_match_start_sec),
        clip2_anchor_sec=float(segment.clip2_match_start_sec),
        offset_sec=float(segment.clip2_match_start_sec - segment.clip1_match_start_sec),
        clip1_start_sec=float(segment.clip1_export_start_sec),
        clip2_start_sec=float(segment.clip2_export_start_sec),
        output_duration_sec=float(output_duration),
        confidence=float(segment.score),
        method=method,
        segments=[segment],
        best_segment_index=0,
    )


def export_segment_video(
    clip1_path: Path,
    clip2_path: Path,
    output_path: Path,
    segment: AlignSegment,
    config: AlignConfig,
    *,
    mute_clip1: bool = False,
    mute_clip2: bool = False,
) -> Path:
    return export_aligned_video(
        clip1_path=clip1_path,
        clip2_path=clip2_path,
        output_path=output_path,
        result=result_from_segment(segment),
        config=config,
        mute_clip1=mute_clip1,
        mute_clip2=mute_clip2,
    )


def export_segment_to_end_video(
    clip1_path: Path,
    clip2_path: Path,
    output_path: Path,
    segment: AlignSegment,
    config: AlignConfig,
    *,
    mute_clip1: bool = False,
    mute_clip2: bool = False,
) -> Path:
    m1 = probe_media(clip1_path)
    m2 = probe_media(clip2_path)
    return export_aligned_video(
        clip1_path=clip1_path,
        clip2_path=clip2_path,
        output_path=output_path,
        result=result_from_segment_to_end(
            segment,
            clip1_duration_sec=float(m1.duration_sec),
            clip2_duration_sec=float(m2.duration_sec),
        ),
        config=config,
        mute_clip1=mute_clip1,
        mute_clip2=mute_clip2,
    )


def export_multi_segment_videos(
    clip1_path: Path,
    clip2_path: Path,
    output_dir: Path,
    result: AlignResult,
    config: AlignConfig,
    *,
    base_name: str,
    mute_clip1: bool = False,
    mute_clip2: bool = False,
) -> list[Path]:
    if not result.segments:
        raise RuntimeError("当前结果不包含可导出的片段")

    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    for segment in result.segments:
        best_tag = "_best" if segment.is_best else ""
        file_name = f"{base_name}_segment_{segment.rank:02d}{best_tag}.mp4"
        out_path = output_dir / file_name
        export_segment_video(
            clip1_path=clip1_path,
            clip2_path=clip2_path,
            output_path=out_path,
            segment=segment,
            config=config,
            mute_clip1=mute_clip1,
            mute_clip2=mute_clip2,
        )
        exported.append(out_path)
    return exported


def export_full_clip_overlay_video(
    clip1_path: Path,
    clip2_path: Path,
    output_path: Path,
    result: AlignResult,
    config: AlignConfig,
) -> Path:
    if not result.segments:
        raise RuntimeError("当前结果不包含可用于整段叠加的片段")

    m1 = probe_media(clip1_path)
    m2 = probe_media(clip2_path)
    if not m1.has_video:
        raise RuntimeError("Clip1 必须是视频文件")
    if not m1.has_audio:
        raise RuntimeError("Clip1 必须包含音轨，才能进行整段叠加导出")
    if not m2.has_audio:
        raise RuntimeError("Clip2 必须包含音轨，才能进行整段叠加导出")

    selected_segments = filter_overlay_segments(result.segments, config.similar_min_segment_gap_sec)
    if not selected_segments:
        raise RuntimeError("未找到满足正序且不重复的叠加段落")

    crf, preset = _preset_crf(config.output_target_preset)
    filter_complex, map_args = _build_full_clip_overlay_filter(
        clip1_duration_sec=float(m1.duration_sec),
        selected_segments=selected_segments,
        config=config,
        out_width=int(config.output_width),
        out_fps=int(config.output_fps),
    )

    def _base(ffmpeg_bin: str) -> list[str]:
        return [
            ffmpeg_bin,
            "-y",
            "-i",
            str(clip1_path),
            "-i",
            str(clip2_path),
            "-filter_complex",
            filter_complex,
            *map_args,
        ]

    tail = [
        "-c:a",
        "aac",
        "-b:a",
        f"{int(config.output_audio_bitrate_k)}k",
        "-movflags",
        "+faststart",
        "-crf",
        str(int(config.output_crf or crf)),
        "-preset",
        str(config.output_preset or preset),
        str(output_path),
    ]

    _run_ffmpeg_with_fallback(
        {
            "base": _base,
            "tail": tail,
            "candidates": _codec_candidates(config.output_video_codec),
        }
    )
    return output_path
