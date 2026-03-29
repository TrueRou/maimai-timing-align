from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks

from .media import probe_media
from .models import AlignConfig, AlignResult, TransitionDebug


@dataclass(slots=True)
class _SignalPack:
    timestamps: np.ndarray
    score: np.ndarray
    peak_time: float
    peak_score: float
    selected_index: int | None = None
    source_frame_indices: np.ndarray | None = None


@dataclass(slots=True)
class _AnchorWeights:
    front_bias_strength: float
    late_penalty_from: float
    late_penalty: float
    prev_pair_boost: float
    next_pair_penalty: float
    first_in_pair_penalty: float
    pair_min_gap_sec: float
    pair_max_gap_sec: float


@dataclass(slots=True)
class _CollapseWeights:
    level_ratio: float
    hold_sec: float
    max_search_sec: float


def _ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _robust_z(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = 1.4826 * mad + 1e-6
    return (x - med) / scale


def _preprocess_frame(frame: np.ndarray, resize_width: int, center_crop_ratio: float) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = resize_width / float(w)
    frame = cv2.resize(frame, (resize_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)

    h2, w2 = frame.shape[:2]
    cw = int(w2 * center_crop_ratio)
    ch = int(h2 * center_crop_ratio)
    x0 = max(0, (w2 - cw) // 2)
    y0 = max(0, (h2 - ch) // 2)
    frame = frame[y0 : y0 + ch, x0 : x0 + cw]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
    return gray


def _build_anchor_weights(config: AlignConfig, role: str) -> _AnchorWeights:
    if role == "clip2":
        return _AnchorWeights(
            front_bias_strength=config.clip2_anchor_front_bias_strength,
            late_penalty_from=config.clip2_anchor_late_penalty_from,
            late_penalty=config.clip2_anchor_late_penalty,
            prev_pair_boost=config.clip2_anchor_prev_pair_boost,
            next_pair_penalty=config.clip2_anchor_next_pair_penalty,
            first_in_pair_penalty=config.clip2_anchor_first_in_pair_penalty,
            pair_min_gap_sec=config.anchor_pair_min_gap_sec,
            pair_max_gap_sec=config.anchor_pair_max_gap_sec,
        )
    return _AnchorWeights(
        front_bias_strength=config.anchor_front_bias_strength,
        late_penalty_from=config.anchor_late_penalty_from,
        late_penalty=config.anchor_late_penalty,
        prev_pair_boost=config.anchor_prev_pair_boost,
        next_pair_penalty=config.anchor_next_pair_penalty,
        first_in_pair_penalty=config.anchor_first_in_pair_penalty,
        pair_min_gap_sec=config.anchor_pair_min_gap_sec,
        pair_max_gap_sec=config.anchor_pair_max_gap_sec,
    )


def _build_collapse_weights(config: AlignConfig, role: str) -> _CollapseWeights:
    if role == "clip2":
        return _CollapseWeights(
            level_ratio=config.clip2_anchor_collapse_level_ratio,
            hold_sec=config.clip2_anchor_collapse_hold_sec,
            max_search_sec=config.clip2_anchor_collapse_max_search_sec,
        )
    return _CollapseWeights(
        level_ratio=config.anchor_collapse_level_ratio,
        hold_sec=config.anchor_collapse_hold_sec,
        max_search_sec=config.anchor_collapse_max_search_sec,
    )


def _select_anchor_peak(
    ts_arr: np.ndarray,
    score: np.ndarray,
    peaks: list[int],
    weights: _AnchorWeights,
    min_peak_z: float,
) -> int:
    if len(peaks) == 1:
        return peaks[0]

    t0 = float(ts_arr[0])
    t1 = float(ts_arr[-1])
    dur = max(1e-6, t1 - t0)

    sorted_peaks = sorted(int(i) for i in peaks)
    best_idx = sorted_peaks[0]
    best_w = -1.0

    for pos, idx in enumerate(sorted_peaks):
        t = float(ts_arr[idx])
        rel = float(np.clip((t - t0) / dur, 0.0, 1.0))

        base = max(0.0, float(score[idx]) - min_peak_z + 1.0)
        front_w = math.exp(-weights.front_bias_strength * rel)
        if rel >= weights.late_penalty_from:
            front_w *= weights.late_penalty

        prev_gap = float("inf")
        next_gap = float("inf")
        if pos > 0:
            prev_gap = t - float(ts_arr[sorted_peaks[pos - 1]])
        if pos + 1 < len(sorted_peaks):
            next_gap = float(ts_arr[sorted_peaks[pos + 1]]) - t

        lo = weights.pair_min_gap_sec
        hi = weights.pair_max_gap_sec
        prev_boost = weights.prev_pair_boost if lo <= prev_gap <= hi else 1.0
        next_penalty = weights.next_pair_penalty if lo <= next_gap <= hi else 1.0
        first_penalty = weights.first_in_pair_penalty if (pos == 0 and lo <= next_gap <= hi) else 1.0

        weighted = base * front_w * prev_boost * next_penalty * first_penalty
        if weighted > best_w:
            best_w = weighted
            best_idx = idx

    return best_idx


def _select_collapse_edge(
    ts_arr: np.ndarray,
    score: np.ndarray,
    peak_idx: int,
    weights: _CollapseWeights,
) -> int:
    n = int(score.size)
    if n <= 2:
        return peak_idx

    dt = float(np.median(np.diff(ts_arr))) if ts_arr.size > 1 else 1.0 / 15.0
    hold_frames = max(1, int(round(weights.hold_sec / max(1e-6, dt))))
    search_end_t = float(ts_arr[peak_idx]) + weights.max_search_sec

    baseline = float(np.median(score))
    peak_v = float(score[peak_idx])
    prominence = max(1e-6, peak_v - baseline)
    level = baseline + prominence * float(np.clip(weights.level_ratio, 0.05, 0.95))

    below_count = 0
    last_high_idx = peak_idx

    for i in range(peak_idx + 1, n):
        if float(ts_arr[i]) > search_end_t:
            break

        if float(score[i]) >= level:
            last_high_idx = i
            below_count = 0
        else:
            below_count += 1
            if below_count >= hold_frames:
                return max(peak_idx, i - below_count)

    return last_high_idx


def extract_transition_signal(video_path: Path, config: AlignConfig, role: str = "clip1") -> _SignalPack:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    source_fps = source_fps if source_fps > 0 else 30.0

    next_t = 0.0
    min_dt = 1.0 / max(1.0, config.sample_fps)

    timestamps: list[float] = []
    source_frame_indices: list[int] = []
    luma_list: list[float] = []
    hist_list: list[float] = []
    edge_list: list[float] = []
    diff_list: list[float] = []

    prev_gray: np.ndarray | None = None
    prev_luma: float | None = None
    prev_hist: np.ndarray | None = None
    prev_edge: float | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        if ts_ms <= 0:
            ts = frame_idx / source_fps
        else:
            ts = ts_ms / 1000.0
        frame_idx += 1

        if ts + 1e-6 < next_t:
            continue
        next_t = ts + min_dt

        gray = _preprocess_frame(frame, config.resize_width, config.center_crop_ratio)
        luma = float(np.median(gray))

        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).astype(np.float32).flatten()
        hist /= float(hist.sum() + 1e-6)

        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.mean(edges > 0))

        if prev_gray is None:
            prev_gray = gray
            prev_luma = luma
            prev_hist = hist
            prev_edge = edge_density
            continue

        if prev_luma is None or prev_hist is None or prev_edge is None:
            prev_gray = gray
            prev_luma = luma
            prev_hist = hist
            prev_edge = edge_density
            continue

        frame_diff = float(np.mean(cv2.absdiff(gray, prev_gray)))
        luma_delta = abs(luma - prev_luma)
        hist_delta = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CHISQR))
        edge_delta = abs(edge_density - prev_edge)

        timestamps.append(ts)
        source_frame_indices.append(frame_idx - 1)
        luma_list.append(luma_delta)
        hist_list.append(hist_delta)
        edge_list.append(edge_delta)
        diff_list.append(frame_diff)

        prev_gray = gray
        prev_luma = luma
        prev_hist = hist
        prev_edge = edge_density

    cap.release()

    if len(timestamps) < 8:
        raise RuntimeError("可用采样帧过少，无法检测转场")

    luma_arr = np.array(luma_list, dtype=np.float32)
    hist_arr = np.array(hist_list, dtype=np.float32)
    edge_arr = np.array(edge_list, dtype=np.float32)
    diff_arr = np.array(diff_list, dtype=np.float32)

    score = _robust_z(luma_arr) + _robust_z(hist_arr) + _robust_z(edge_arr) + _robust_z(diff_arr)

    k = max(3, _ensure_odd(config.smoothing_window))
    kernel = np.ones(k, dtype=np.float32) / float(k)
    score = np.convolve(score, kernel, mode="same")

    ts_arr = np.array(timestamps, dtype=np.float64)
    dt = float(np.median(np.diff(ts_arr))) if ts_arr.size > 1 else (1.0 / config.sample_fps)
    min_distance = max(1, int(config.min_peak_distance_sec / max(1e-6, dt)))

    peaks, _ = find_peaks(score, distance=min_distance)
    peaks = [int(i) for i in peaks if score[i] >= config.min_peak_z]
    if not peaks:
        idx = int(np.argmax(score))
        if score[idx] < config.min_peak_z * 0.7:
            raise RuntimeError("未检测到足够明显的转场")
        peaks = [idx]

    weights = _build_anchor_weights(config, role)
    best_peak = _select_anchor_peak(ts_arr, score, peaks, weights, config.min_peak_z)
    best = best_peak
    return _SignalPack(
        timestamps=ts_arr,
        score=score.astype(np.float64),
        peak_time=float(ts_arr[best]),
        peak_score=float(score[best_peak]),
        selected_index=int(best),
        source_frame_indices=np.array(source_frame_indices, dtype=np.int32),
    )


def _interp_signal(ts: np.ndarray, score: np.ndarray, t0: float, t1: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if t1 <= t0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    grid = np.arange(t0, t1, dt, dtype=np.float64)
    if grid.size < 8:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    y = np.interp(grid, ts, score)
    return grid, y


def _corr_on_interval(
    s1: _SignalPack,
    s2: _SignalPack,
    offset_sec: float,
    t_start: float,
    t_end: float,
    dt: float,
    min_overlap_sec: float,
) -> tuple[float, float]:
    if t_end <= t_start:
        return float("nan"), 0.0

    grid = np.arange(t_start, t_end, dt, dtype=np.float64)
    if grid.size < 8:
        return float("nan"), 0.0

    y1 = np.interp(grid, s1.timestamps, s1.score, left=np.nan, right=np.nan)
    y2 = np.interp(grid + offset_sec, s2.timestamps, s2.score, left=np.nan, right=np.nan)
    valid = np.isfinite(y1) & np.isfinite(y2)

    overlap = float(valid.sum()) * dt
    if overlap < min_overlap_sec or valid.sum() < 8:
        return float("nan"), overlap

    a = y1[valid]
    b = y2[valid]
    a = (a - a.mean()) / (a.std() + 1e-6)
    b = (b - b.mean()) / (b.std() + 1e-6)
    corr = float(np.mean(a * b))
    return (corr if math.isfinite(corr) else float("nan")), overlap


def _score_offset_global(
    s1: _SignalPack,
    s2: _SignalPack,
    offset_sec: float,
    config: AlignConfig,
) -> float:
    t1_min = float(s1.timestamps[0])
    t1_max = float(s1.timestamps[-1])
    t2_min = float(s2.timestamps[0]) - offset_sec
    t2_max = float(s2.timestamps[-1]) - offset_sec

    overlap_start = max(t1_min, t2_min)
    overlap_end = min(t1_max, t2_max)
    overlap = overlap_end - overlap_start
    if overlap < config.global_match_min_overlap_sec:
        return -1.0

    dt = max(0.005, min(0.05, config.global_scan_step_sec))
    win = min(config.global_match_window_sec, overlap)
    if win <= 0.0:
        return -1.0

    mid = 0.5 * (overlap_start + overlap_end)
    half_win = 0.5 * win
    sub_overlap = max(4.0, min(win * 0.6, config.global_match_min_overlap_sec))

    windows = [
        (0.40, overlap_start, overlap_end, config.global_match_min_overlap_sec),
        (0.20, overlap_start, overlap_start + win, sub_overlap),
        (0.20, mid - half_win, mid + half_win, sub_overlap),
        (0.20, overlap_end - win, overlap_end, sub_overlap),
    ]

    weighted_sum = 0.0
    weight_total = 0.0
    for weight, a, b, need_overlap in windows:
        corr, _ = _corr_on_interval(s1, s2, offset_sec, a, b, dt, need_overlap)
        if math.isfinite(corr):
            weighted_sum += weight * corr
            weight_total += weight

    if weight_total <= 1e-6:
        return -1.0
    return weighted_sum / weight_total


def _estimate_global_offset(
    s1: _SignalPack,
    s2: _SignalPack,
    offset_transition: float,
    config: AlignConfig,
) -> tuple[float, float]:
    step = max(0.005, float(config.global_scan_step_sec))
    span = max(step, float(config.global_search_range_sec))
    offsets = np.arange(offset_transition - span, offset_transition + span + step * 0.5, step)

    best_offset = offset_transition
    best_score = -2.0
    scored: list[tuple[float, float]] = []

    for off in offsets:
        score = _score_offset_global(s1, s2, float(off), config)
        if score <= -1.0:
            continue
        scored.append((float(off), float(score)))
        if score > best_score:
            best_score = float(score)
            best_offset = float(off)

    if not scored:
        return offset_transition, 0.0

    # 局部抛物线细化
    best_idx = min(range(len(scored)), key=lambda i: abs(scored[i][0] - best_offset))
    if 0 < best_idx < len(scored) - 1:
        l_off, l_score = scored[best_idx - 1]
        c_off, c_score = scored[best_idx]
        r_off, r_score = scored[best_idx + 1]
        if abs((c_off - l_off) - step) < 1e-6 and abs((r_off - c_off) - step) < 1e-6:
            denom = l_score - 2.0 * c_score + r_score
            if abs(denom) > 1e-9:
                delta = 0.5 * (l_score - r_score) / denom
                delta = float(np.clip(delta, -1.0, 1.0))
                refined = c_off + delta * step
                refined_score = _score_offset_global(s1, s2, refined, config)
                if refined_score > best_score:
                    best_score = refined_score
                    best_offset = refined

    base_conf = float(np.clip((best_score + 1.0) / 2.0, 0.0, 1.0))
    if len(scored) >= 2:
        top2 = sorted((x[1] for x in scored), reverse=True)[:2]
        margin_conf = float(np.clip((top2[0] - top2[1]) / 0.25, 0.0, 1.0))
    else:
        margin_conf = 0.0
    global_conf = float(np.clip(0.8 * base_conf + 0.2 * margin_conf, 0.0, 1.0))
    return best_offset, global_conf


def _refine_offset(
    s1: _SignalPack,
    s2: _SignalPack,
    offset0: float,
    window_sec: float = 12.0,
    max_shift_sec: float = 0.8,
    dt: float = 0.01,
) -> tuple[float, float]:
    t1_start = s1.peak_time
    t2_start = s2.peak_time

    _, y1 = _interp_signal(s1.timestamps, s1.score, t1_start, t1_start + window_sec, dt)
    _, y2 = _interp_signal(s2.timestamps, s2.score, t2_start, t2_start + window_sec, dt)
    if y1.size < 16 or y2.size < 16:
        return offset0, 0.0

    n = min(y1.size, y2.size)
    y1 = y1[:n]
    y2 = y2[:n]
    y1 = (y1 - y1.mean()) / (y1.std() + 1e-6)
    y2 = (y2 - y2.mean()) / (y2.std() + 1e-6)

    shifts = np.arange(-max_shift_sec, max_shift_sec + dt, dt, dtype=np.float64)
    best_corr = -2.0
    best_shift = 0.0

    base_t = np.arange(0, n, dtype=np.float64) * dt

    for sh in shifts:
        shifted = np.interp(base_t + sh, base_t, y2, left=np.nan, right=np.nan)
        valid = np.isfinite(shifted)
        if valid.sum() < n * 0.7:
            continue
        corr = float(np.corrcoef(y1[valid], shifted[valid])[0, 1])
        if math.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_shift = float(sh)

    if best_corr < -1.0:
        return offset0, 0.0

    return offset0 + best_shift, float((best_corr + 1.0) / 2.0)


def align_videos(
    clip1_path: Path,
    clip2_path: Path,
    config: AlignConfig,
    manual_anchor1: float | None = None,
    manual_anchor2: float | None = None,
) -> AlignResult:
    m1 = probe_media(clip1_path)
    m2 = probe_media(clip2_path)

    s1 = extract_transition_signal(clip1_path, config, role="clip1")
    s2 = extract_transition_signal(clip2_path, config, role="clip2")

    anchor1 = float(manual_anchor1) if manual_anchor1 is not None else s1.peak_time
    anchor2 = float(manual_anchor2) if manual_anchor2 is not None else s2.peak_time

    offset_transition = anchor2 - anchor1
    global_offset, global_conf = _estimate_global_offset(s1, s2, offset_transition, config)

    if global_conf >= config.global_confidence_floor:
        offset_seed = global_offset
    else:
        w = float(np.clip(config.global_low_conf_global_weight, 0.0, 1.0))
        offset_seed = (1.0 - w) * offset_transition + w * global_offset

    offset, corr_conf = _refine_offset(
        s1,
        s2,
        offset0=offset_seed,
        max_shift_sec=config.global_refine_radius_sec,
    )

    clip1_start = anchor1
    clip2_start = clip1_start + offset

    if clip2_start < 0:
        clip1_start -= clip2_start
        clip2_start = 0.0

    dur1 = m1.duration_sec - clip1_start
    dur2 = m2.duration_sec - clip2_start
    output_duration = min(dur1, dur2)
    if output_duration <= 0.5:
        raise RuntimeError("锚点后无足够重叠时长，无法导出")

    peak_score_mean = max(0.0, (s1.peak_score + s2.peak_score) / 2.0)
    peak_conf = float(np.clip((peak_score_mean - config.min_peak_z + 1.0) / 4.0, 0.0, 1.0))
    confidence = float(np.clip(0.35 * peak_conf + 0.35 * corr_conf + 0.30 * global_conf, 0.0, 1.0))

    return AlignResult(
        clip1_anchor_sec=anchor1,
        clip2_anchor_sec=anchor2,
        offset_sec=offset,
        clip1_start_sec=clip1_start,
        clip2_start_sec=clip2_start,
        output_duration_sec=output_duration,
        confidence=confidence,
        debug1=TransitionDebug(
            timestamps=s1.timestamps.tolist(),
            score=s1.score.tolist(),
            peak_time=s1.peak_time,
            peak_score=s1.peak_score,
            selected_index=s1.selected_index,
            source_frame_indices=s1.source_frame_indices.tolist() if s1.source_frame_indices is not None else None,
        ),
        debug2=TransitionDebug(
            timestamps=s2.timestamps.tolist(),
            score=s2.score.tolist(),
            peak_time=s2.peak_time,
            peak_score=s2.peak_score,
            selected_index=s2.selected_index,
            source_frame_indices=s2.source_frame_indices.tolist() if s2.source_frame_indices is not None else None,
        ),
    )
