from __future__ import annotations

import numpy as np

from maimai_timing_align.analysis import (
    _build_anchor_weights,
    _build_collapse_weights,
    _estimate_global_offset,
    _refine_offset,
    _robust_z,
    _select_anchor_peak,
    _select_collapse_edge,
    _SignalPack,
)
from maimai_timing_align.models import AlignConfig


def test_robust_z_centered() -> None:
    arr = np.array([1, 2, 3, 4, 100], dtype=np.float64)
    z = _robust_z(arr)
    assert abs(float(np.median(z))) < 1e-6


def test_refine_offset_with_synthetic_shift() -> None:
    ts = np.arange(0, 12, 0.01, dtype=np.float64)
    y1 = np.sin(ts * 2.0) + 0.25 * np.cos(ts * 5.0)
    shift = 0.17
    y2 = np.sin((ts + shift) * 2.0) + 0.25 * np.cos((ts + shift) * 5.0)

    s1 = _SignalPack(timestamps=ts, score=y1, peak_time=0.0, peak_score=5.0)
    s2 = _SignalPack(timestamps=ts, score=y2, peak_time=0.0, peak_score=5.0)

    refined, conf = _refine_offset(s1, s2, offset0=0.0, window_sec=10.0, max_shift_sec=0.5, dt=0.01)
    assert abs(refined + shift) < 0.03
    assert conf > 0.7


def test_select_anchor_peak_prefers_front_cutout() -> None:
    cfg = AlignConfig()
    ts = np.linspace(0, 100, 1001, dtype=np.float64)
    score = np.zeros_like(ts)

    # 前段切入(较早)
    score[40] = 5.2
    # 前段切出(目标)
    score[52] = 5.1
    # 尾段切出(原始分数更高，若无权重可能误选)
    score[920] = 7.8
    # 尾段切入
    score[935] = 6.1

    peaks = [40, 52, 920, 935]
    idx = _select_anchor_peak(ts, score, peaks, _build_anchor_weights(cfg, "clip1"), cfg.min_peak_z)
    assert idx == 52


def test_select_anchor_peak_clip2_prefers_second_peak_in_early_pair() -> None:
    cfg = AlignConfig()
    ts = np.linspace(0, 120, 1201, dtype=np.float64)
    score = np.zeros_like(ts)

    # clip2 开头通常是切入+切出，目标应选第二峰（切出）
    score[30] = 7.4
    score[43] = 6.6
    # 后续仍有强峰，避免误选
    score[900] = 8.0
    peaks = [30, 43, 900]

    idx = _select_anchor_peak(ts, score, peaks, _build_anchor_weights(cfg, "clip2"), cfg.min_peak_z)
    assert idx == 43


def test_select_collapse_edge_uses_end_of_peak_not_start() -> None:
    cfg = AlignConfig()
    ts = np.arange(0.0, 4.0, 0.1, dtype=np.float64)
    score = np.zeros_like(ts)

    # 峰在1.0s附近开始，平台持续到1.4s，1.5s后崩塌
    score[10:15] = 8.0
    score[15] = 1.6
    score[16] = 0.8

    peak_idx = 10
    idx = _select_collapse_edge(ts, score, peak_idx, _build_collapse_weights(cfg, "clip1"))
    assert idx == 14


def test_estimate_global_offset_recovers_from_wrong_transition_prior() -> None:
    ts = np.arange(0.0, 120.0, 0.02, dtype=np.float64)
    base = (
        np.sin(ts * 0.21 + 0.002 * ts * ts)
        + 0.55 * np.sin(ts * 0.73)
        + 0.25 * np.cos(ts * 1.37)
        + 0.70 * np.exp(-0.5 * ((ts - 24.0) / 1.2) ** 2)
        + 0.55 * np.exp(-0.5 * ((ts - 63.0) / 1.8) ** 2)
        + 0.65 * np.exp(-0.5 * ((ts - 95.0) / 1.3) ** 2)
    )

    real_shift = 3.4
    expected_offset = -real_shift
    y1 = base
    y2 = np.interp(ts + real_shift, ts, y1, left=0.0, right=0.0)
    y2 += 1.8 * np.exp(-0.5 * ((ts - 5.0) / 0.25) ** 2)  # 干扰早峰

    s1 = _SignalPack(timestamps=ts, score=y1, peak_time=30.0, peak_score=7.0)
    s2 = _SignalPack(timestamps=ts, score=y2, peak_time=36.0, peak_score=8.0)

    cfg = AlignConfig(
        global_search_range_sec=20.0,
        global_scan_step_sec=0.02,
        global_match_min_overlap_sec=20.0,
        global_match_window_sec=18.0,
    )

    # 转场先验故意偏到 +6s，全局匹配应拉回到真实偏移
    best_offset, global_conf = _estimate_global_offset(s1, s2, offset_transition=6.0, config=cfg)
    assert abs(best_offset - expected_offset) < 0.10
    assert global_conf > 0.55
