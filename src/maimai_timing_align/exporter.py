from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .media import ensure_ffmpeg_available, probe_media
from .models import AlignConfig, AlignResult


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
