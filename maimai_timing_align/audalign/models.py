from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AudioAlignParams:
    sample_rate: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    search_range_sec: float = 20.0
    min_overlap_sec: float = 15.0
    confidence_floor: float = 0.35
    max_duration_sec: float | None = 300.0
    method: str = "onset_xcorr"


@dataclass(slots=True)
class AudioAlignDiagnostics:
    hop_sec_effective: float
    onset_frames_a: int
    onset_frames_b: int


@dataclass(slots=True)
class AudioAlignResult:
    anchor_a_sec: float
    anchor_b_sec: float
    offset_sec: float
    start_a_sec: float
    start_b_sec: float
    overlap_duration_sec: float
    confidence: float
    method: str
    warnings: list[str] = field(default_factory=list)
    diagnostics: AudioAlignDiagnostics | None = None
