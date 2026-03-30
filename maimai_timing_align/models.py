from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class AlignConfig:
    otoge_base_url: str = "http://127.0.0.1:8000"
    otoge_developer_token: str = ""
    otoge_timeout_sec: float = 30.0
    align_backend: Literal["remote", "local"] = "remote"
    align_mode: Literal["standard", "similar_segments"] = "standard"
    align_fallback_to_local: bool = True
    audio_sr: int = 22050
    audio_hop_length: int = 512
    audio_n_fft: int = 2048
    audio_search_range_sec: float = 20.0
    audio_min_overlap_sec: float = 15.0
    audio_confidence_floor: float = 0.35
    audio_max_duration_sec: float = 300.0
    similar_match_window_sec: float = 12.0
    similar_match_step_sec: float = 2.0
    similar_similarity_floor: float = 0.58
    similar_max_segments: int = 6
    similar_min_segment_gap_sec: float = 5.0
    similar_margin_before_sec: float = 1.5
    similar_margin_after_sec: float = 2.0
    similar_onset_weight: float = 0.4
    similar_chroma_weight: float = 0.3
    similar_tempogram_weight: float = 0.3
    similar_num_workers: int = 0
    similar_export_mode: Literal["segment_exports", "full_clip_overlay"] = "segment_exports"
    osu_client_id: str = ""
    osu_client_secret: str = ""
    osu_query: str = ""
    osu_artist: str = ""
    osu_creator: str = ""
    osu_version: str = ""
    osu_bpm: float = 0.0
    osu_mode: Literal["any", "osu", "taiko", "fruits", "mania"] = "any"
    osu_category: Literal["has_leaderboard", "ranked", "loved", "qualified", "pending", "graveyard"] = "has_leaderboard"
    osu_batch_limit: int = 5
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
class AlignSegment:
    segment_id: str
    rank: int
    is_best: bool
    clip1_match_start_sec: float
    clip2_match_start_sec: float
    match_duration_sec: float
    clip1_export_start_sec: float
    clip2_export_start_sec: float
    export_duration_sec: float
    score: float
    onset_score: float
    chroma_score: float
    tempogram_score: float
    note: str | None = None

    @property
    def clip1_match_end_sec(self) -> float:
        return float(self.clip1_match_start_sec + self.match_duration_sec)

    @property
    def clip2_match_end_sec(self) -> float:
        return float(self.clip2_match_start_sec + self.match_duration_sec)

    @property
    def clip1_export_end_sec(self) -> float:
        return float(self.clip1_export_start_sec + self.export_duration_sec)

    @property
    def clip2_export_end_sec(self) -> float:
        return float(self.clip2_export_start_sec + self.export_duration_sec)


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
    segments: list[AlignSegment] | None = None
    best_segment_index: int = 0

    @property
    def best_segment(self) -> AlignSegment | None:
        if not self.segments:
            return None
        idx = min(max(int(self.best_segment_index), 0), len(self.segments) - 1)
        return self.segments[idx]


@dataclass(slots=True)
class OsuBeatmapCandidate:
    beatmap_id: int
    beatmapset_id: int
    artist: str
    title: str
    version: str
    creator: str
    bpm: float
    mode: str
    source_url: str


@dataclass(slots=True)
class OsuBatchMatchResult:
    candidate: OsuBeatmapCandidate
    audio_path: Path
    result: AlignResult
