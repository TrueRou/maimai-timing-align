from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class MediaInfo:
    duration_sec: float
    has_audio: bool
    has_video: bool


@dataclass(slots=True)
class FFmpegBinaries:
    ffmpeg: str
    ffprobe: str


def _resolve_bin_name(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _resolve_bundled_bin(name: str) -> str | None:
    base_dir = getattr(sys, "_MEIPASS", None)
    if not base_dir:
        return None
    p = Path(base_dir) / "bin" / _resolve_bin_name(name)
    if p.exists():
        return str(p)
    return None


def resolve_ffmpeg_binaries() -> FFmpegBinaries:
    bundled_ffmpeg = _resolve_bundled_bin("ffmpeg")
    bundled_ffprobe = _resolve_bundled_bin("ffprobe")
    ffmpeg = bundled_ffmpeg or shutil.which("ffmpeg")
    ffprobe = bundled_ffprobe or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("未找到 ffmpeg/ffprobe，请先安装或使用打包版可执行程序")
    return FFmpegBinaries(ffmpeg=ffmpeg, ffprobe=ffprobe)


def ensure_ffmpeg_available() -> FFmpegBinaries:
    return resolve_ffmpeg_binaries()


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "命令执行失败").strip()[-4000:])
    return proc


def probe_media(path: Path) -> MediaInfo:
    bins = ensure_ffmpeg_available()
    cmd = [
        bins.ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-show_streams",
        "-of",
        "json",
        str(path),
    ]
    proc = _run_command(cmd)

    payload = json.loads(proc.stdout or "{}")
    duration = float(payload.get("format", {}).get("duration", 0.0) or 0.0)
    has_audio = False
    has_video = False
    for stream in payload.get("streams", []):
        codec_type = stream.get("codec_type")
        if codec_type == "audio":
            has_audio = True
        elif codec_type == "video":
            has_video = True

    if duration <= 0:
        raise RuntimeError(f"无法读取媒体时长: {path}")

    return MediaInfo(duration_sec=duration, has_audio=has_audio, has_video=has_video)


def extract_audio_track(input_path: Path, output_wav_path: Path) -> Path:
    bins = ensure_ffmpeg_available()
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        bins.ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "44100",
        "-c:a",
        "pcm_s16le",
        str(output_wav_path),
    ]
    _run_command(cmd)
    return output_wav_path
