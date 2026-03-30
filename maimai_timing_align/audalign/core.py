import numpy as np


def _z_norm(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sigma = float(np.std(x)) + 1e-6
    return (x - mu) / sigma


def _compute_shift_correlation(a: np.ndarray, b: np.ndarray, shift: int) -> tuple[float, int]:
    n = min(a.size, b.size)
    if n < 8:
        return -1.0, 0

    if shift >= 0:
        if shift >= n:
            return -1.0, 0
        x = a[: n - shift]
        y = b[shift:n]
    else:
        sh = -shift
        if sh >= n:
            return -1.0, 0
        x = a[sh:n]
        y = b[: n - sh]

    if x.size < 8:
        return -1.0, 0

    x = _z_norm(x)
    y = _z_norm(y)
    corr = float(np.mean(x * y))
    if not np.isfinite(corr):
        return -1.0, x.size
    return corr, x.size


def estimate_offset_from_onset_vectors(
    onset_a: np.ndarray,
    onset_b: np.ndarray,
    hop_sec: float,
    search_range_sec: float,
    min_overlap_sec: float,
) -> tuple[float, float]:
    if onset_a.size < 8 or onset_b.size < 8:
        return 0.0, 0.0

    if float(np.std(onset_a)) < 1e-8 and float(np.std(onset_b)) < 1e-8:
        return 0.0, 0.0

    a = _z_norm(onset_a.astype(np.float64))
    b = _z_norm(onset_b.astype(np.float64))

    max_shift = int(round(search_range_sec / max(1e-6, hop_sec)))
    min_overlap = int(round(min_overlap_sec / max(1e-6, hop_sec)))

    scored: list[tuple[int, float]] = []
    best_shift = 0
    best_corr = -1.0

    for sh in range(-max_shift, max_shift + 1):
        corr, overlap = _compute_shift_correlation(a, b, sh)
        if overlap < max(8, min_overlap):
            continue
        scored.append((sh, corr))
        if corr > best_corr + 1e-12:
            best_corr = corr
            best_shift = sh
        elif abs(corr - best_corr) <= 1e-12 and abs(sh) < abs(best_shift):
            best_shift = sh

    if not scored:
        return 0.0, 0.0

    scored.sort(key=lambda x: x[0])
    idx = min(range(len(scored)), key=lambda i: abs(scored[i][0] - best_shift))
    best_shift_refined = float(best_shift)
    if 0 < idx < len(scored) - 1:
        l_sh, l_c = scored[idx - 1]
        c_sh, c_c = scored[idx]
        r_sh, r_c = scored[idx + 1]
        if l_sh + 1 == c_sh and c_sh + 1 == r_sh:
            denom = l_c - 2.0 * c_c + r_c
            if abs(denom) > 1e-9:
                delta = 0.5 * (l_c - r_c) / denom
                delta = float(np.clip(delta, -1.0, 1.0))
                best_shift_refined = float(c_sh) + delta

    # 正值表示 B 相对 A 更晚：offset = b - a
    offset_sec = best_shift_refined * hop_sec

    top = sorted((c for _, c in scored), reverse=True)
    c1 = float(np.clip((best_corr + 1.0) / 2.0, 0.0, 1.0))
    c2 = float(np.clip((top[0] - top[1]) / 0.2, 0.0, 1.0)) if len(top) >= 2 else 0.0
    conf = float(np.clip(0.8 * c1 + 0.2 * c2, 0.0, 1.0))
    return offset_sec, conf
