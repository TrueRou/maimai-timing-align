from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AlignConfig:
    otoge_base_url: str = "http://127.0.0.1:8000"
    otoge_developer_token: str = ""
    otoge_timeout_sec: float = 30.0
    audio_sr: int = 22050
    audio_hop_length: int = 512
    audio_n_fft: int = 2048
    audio_search_range_sec: float = 20.0
    audio_min_overlap_sec: float = 15.0
    audio_confidence_floor: float = 0.35
    audio_max_duration_sec: float = 300.0
    audio1_gain_db: float = 0.0
    audio2_gain_db: float = -6.0
    audio_reverb_wet: float = 0.12
    output_video_codec: str = "h265"
    output_target_preset: str = "balanced"
    output_crf: int | None = None
    output_preset: str | None = None
    output_width: int = 1080
    output_fps: int = 30
    output_audio_bitrate_k: int = 128
    preview_duration_sec: float = 12.0
    preview_video_bitrate_k: int = 3500


@dataclass(slots=True)
class AlignResult:
    clip1_anchor_sec: float
    clip2_anchor_sec: float
    offset_sec: float
    clip1_start_sec: float
    clip2_start_sec: float
    output_duration_sec: float
    confidence: float
    method: str = "audio_remote"
    warnings: list[str] | None = None
    output_path: Path | None = None
