from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AlignConfig:
    sample_fps: float = 15.0
    resize_width: int = 640
    center_crop_ratio: float = 0.82
    smoothing_window: int = 7
    min_peak_distance_sec: float = 0.35
    min_peak_z: float = 2.8
    anchor_front_bias_strength: float = 3.4
    anchor_late_penalty_from: float = 0.65
    anchor_late_penalty: float = 0.35
    anchor_prev_pair_boost: float = 1.25
    anchor_next_pair_penalty: float = 0.92
    anchor_first_in_pair_penalty: float = 1.0
    anchor_pair_min_gap_sec: float = 0.15
    anchor_pair_max_gap_sec: float = 2.50
    anchor_collapse_level_ratio: float = 0.40
    anchor_collapse_hold_sec: float = 0.10
    anchor_collapse_max_search_sec: float = 2.00
    clip2_anchor_front_bias_strength: float = 2.0
    clip2_anchor_late_penalty_from: float = 0.75
    clip2_anchor_late_penalty: float = 0.45
    clip2_anchor_prev_pair_boost: float = 1.65
    clip2_anchor_next_pair_penalty: float = 0.90
    clip2_anchor_first_in_pair_penalty: float = 0.62
    clip2_anchor_collapse_level_ratio: float = 0.50
    clip2_anchor_collapse_hold_sec: float = 0.08
    clip2_anchor_collapse_max_search_sec: float = 1.60
    global_search_range_sec: float = 20.0
    global_scan_step_sec: float = 0.02
    global_match_min_overlap_sec: float = 20.0
    global_match_window_sec: float = 18.0
    global_low_conf_global_weight: float = 0.25
    global_confidence_floor: float = 0.45
    global_refine_radius_sec: float = 1.0
    audio2_gain_db: float = -6.0
    output_crf: int = 18
    output_preset: str = "medium"


@dataclass(slots=True)
class TransitionDebug:
    timestamps: list[float]
    score: list[float]
    peak_time: float | None
    peak_score: float | None
    selected_index: int | None = None
    source_frame_indices: list[int] | None = None


@dataclass(slots=True)
class AlignResult:
    clip1_anchor_sec: float
    clip2_anchor_sec: float
    offset_sec: float
    clip1_start_sec: float
    clip2_start_sec: float
    output_duration_sec: float
    confidence: float
    output_path: Path | None = None
    debug1: TransitionDebug | None = None
    debug2: TransitionDebug | None = None
