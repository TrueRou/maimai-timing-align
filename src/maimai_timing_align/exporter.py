from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from .media import probe_media
from .models import AlignConfig, AlignResult


def _fmt(v: float) -> str:
    return f"{v:.6f}"


def export_aligned_video(
    clip1_path: Path,
    clip2_path: Path,
    output_path: Path,
    result: AlignResult,
    config: AlignConfig,
) -> Path:
    m1 = probe_media(clip1_path)
    m2 = probe_media(clip2_path)

    s1 = result.clip1_start_sec
    s2 = result.clip2_start_sec
    d = result.output_duration_sec

    filter_parts: list[str] = [
        f"[0:v]trim=start={_fmt(s1)}:duration={_fmt(d)},setpts=PTS-STARTPTS[vout]"
    ]
    map_args = ["-map", "[vout]"]

    if m1.has_audio and m2.has_audio:
        filter_parts.extend(
            [
                f"[0:a]atrim=start={_fmt(s1)}:duration={_fmt(d)},asetpts=PTS-STARTPTS[a1]",
                f"[1:a]atrim=start={_fmt(s2)}:duration={_fmt(d)},asetpts=PTS-STARTPTS,volume={config.audio2_gain_db}dB[a2]",
                "[a1][a2]amix=inputs=2:normalize=0:dropout_transition=0,alimiter=limit=0.95[aout]",
            ]
        )
        map_args.extend(["-map", "[aout]"])
    elif m1.has_audio:
        filter_parts.append(
            f"[0:a]atrim=start={_fmt(s1)}:duration={_fmt(d)},asetpts=PTS-STARTPTS[aout]"
        )
        map_args.extend(["-map", "[aout]"])
    elif m2.has_audio:
        filter_parts.append(
            f"[1:a]atrim=start={_fmt(s2)}:duration={_fmt(d)},asetpts=PTS-STARTPTS[aout]"
        )
        map_args.extend(["-map", "[aout]"])

    filter_complex = ";".join(filter_parts)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(clip1_path),
        "-i",
        str(clip2_path),
        "-filter_complex",
        filter_complex,
        *map_args,
        "-c:v",
        "libx264",
        "-preset",
        config.output_preset,
        "-crf",
        str(config.output_crf),
    ]

    if m1.has_audio or m2.has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.extend(["-an"])

    cmd.append(str(output_path))

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        safe_cmd = " ".join(shlex.quote(x) for x in cmd)
        raise RuntimeError(
            "导出失败。\n"
            f"cmd: {safe_cmd}\n"
            f"stderr: {proc.stderr[-4000:]}"
        )

    return output_path
