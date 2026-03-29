from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MediaInfo:
    duration_sec: float
    has_audio: bool


def ensure_ffmpeg_available() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("未找到 ffmpeg/ffprobe，请先安装并加入 PATH")


def probe_media(path: Path) -> MediaInfo:
    ensure_ffmpeg_available()
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe 执行失败: {proc.stderr.strip()}")

    payload = json.loads(proc.stdout or "{}")
    duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    has_audio = False
    for stream in payload.get("streams", []):
        if stream.get("codec_type") == "audio":
            has_audio = True
            break

    if duration <= 0:
        raise RuntimeError(f"无法读取视频时长: {path}")

    return MediaInfo(duration_sec=duration, has_audio=has_audio)
