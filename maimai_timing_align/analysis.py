from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import librosa
import numpy as np

try:
    from .api import OtogeAlignClient
    from .media import extract_audio_track, probe_media
    from .models import AlignConfig, AlignResult, AlignSegment
except ImportError:  # pragma: no cover
    from api import OtogeAlignClient
    from media import extract_audio_track, probe_media
    from models import AlignConfig, AlignResult, AlignSegment

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _z_norm(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.size == 0:
        return arr
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    if sigma < 1e-8:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - mu) / sigma


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return 0.0
    na = float(np.linalg.norm(aa))
    nb = float(np.linalg.norm(bb))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return _clamp01((float(np.dot(aa, bb) / (na * nb)) + 1.0) / 2.0)


def _profile_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = _z_norm(a)
    bb = _z_norm(b)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return 0.0
    return _clamp01((float(np.mean(aa * bb)) + 1.0) / 2.0)


def _extract_similarity_features(audio_path: Path, config: AlignConfig) -> tuple[dict[str, np.ndarray], float, float]:
    y, sr = librosa.load(
        str(audio_path),
        sr=int(config.audio_sr),
        mono=True,
        duration=float(config.audio_max_duration_sec),
    )
    if y.size < 128:
        raise RuntimeError(f"音频过短或无法读取: {audio_path}")

    hop_length = int(config.audio_hop_length)
    n_fft = int(config.audio_n_fft)
    onset = np.asarray(
        librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=hop_length,
            n_fft=n_fft,
            aggregate=np.median,
        ),
        dtype=np.float64,
    )
    chroma = np.asarray(
        librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            hop_length=hop_length,
            n_fft=n_fft,
        ),
        dtype=np.float64,
    )
    tempogram = np.asarray(
        librosa.feature.tempogram(
            onset_envelope=onset,
            sr=sr,
            hop_length=hop_length,
        ),
        dtype=np.float64,
    )
    hop_sec = float(hop_length) / float(sr)
    duration_sec = float(y.size) / float(sr)
    return {"onset": onset, "chroma": chroma, "tempogram": tempogram}, hop_sec, duration_sec


def _weighted_similarity(
    feature_a: dict[str, np.ndarray],
    feature_b: dict[str, np.ndarray],
    start_a: int,
    start_b: int,
    length_frames: int,
    config: AlignConfig,
) -> tuple[float, float, float, float]:
    onset_a = feature_a["onset"][start_a : start_a + length_frames]
    onset_b = feature_b["onset"][start_b : start_b + length_frames]
    chroma_a = feature_a["chroma"][:, start_a : start_a + length_frames]
    chroma_b = feature_b["chroma"][:, start_b : start_b + length_frames]
    temp_a = feature_a["tempogram"][:, start_a : start_a + length_frames]
    temp_b = feature_b["tempogram"][:, start_b : start_b + length_frames]

    onset_score = _profile_similarity(onset_a, onset_b)
    chroma_score = _cosine_similarity(np.mean(chroma_a, axis=1), np.mean(chroma_b, axis=1))
    temp_score = _cosine_similarity(np.mean(temp_a, axis=1), np.mean(temp_b, axis=1))

    total_weight = (
        float(config.similar_onset_weight)
        + float(config.similar_chroma_weight)
        + float(config.similar_tempogram_weight)
    )
    if total_weight <= 1e-8:
        total_weight = 1.0
    score = (
        float(config.similar_onset_weight) * onset_score
        + float(config.similar_chroma_weight) * chroma_score
        + float(config.similar_tempogram_weight) * temp_score
    ) / total_weight
    return _clamp01(score), onset_score, chroma_score, temp_score


def _select_similar_segments(a1: Path, a2: Path, m1, m2, config: AlignConfig) -> AlignResult:
    features1, hop_sec_1, duration1 = _extract_similarity_features(a1, config)
    features2, hop_sec_2, duration2 = _extract_similarity_features(a2, config)
    hop_sec = min(hop_sec_1, hop_sec_2)

    frames1 = int(features1["onset"].shape[-1])
    frames2 = int(features2["onset"].shape[-1])
    window_frames = max(8, int(round(float(config.similar_match_window_sec) / max(hop_sec, 1e-6))))
    step_frames = max(1, int(round(float(config.similar_match_step_sec) / max(hop_sec, 1e-6))))
    max_frames = min(frames1, frames2)
    if max_frames < window_frames:
        raise RuntimeError("音频太短，无法进行相似段匹配")

    candidates: list[AlignSegment] = []
    floor = float(config.similar_similarity_floor)
    min_gap = max(0.0, float(config.similar_min_segment_gap_sec))
    margin_before = max(0.0, float(config.similar_margin_before_sec))
    margin_after = max(0.0, float(config.similar_margin_after_sec))
    max_segments = max(1, int(config.similar_max_segments))

    candidate_rows: list[tuple[float, float, float, float, float, float]] = []
    for start_a in range(0, frames1 - window_frames + 1, step_frames):
        for start_b in range(0, frames2 - window_frames + 1, step_frames):
            score, onset_score, chroma_score, temp_score = _weighted_similarity(
                features1,
                features2,
                start_a,
                start_b,
                window_frames,
                config,
            )
            if score < floor:
                continue
            candidate_rows.append(
                (
                    score,
                    onset_score,
                    chroma_score,
                    temp_score,
                    float(start_a) * hop_sec,
                    float(start_b) * hop_sec,
                )
            )

    candidate_rows.sort(key=lambda item: item[0], reverse=True)
    for score, onset_score, chroma_score, temp_score, clip1_match_start, clip2_match_start in candidate_rows:
        if any(abs(existing.clip1_match_start_sec - clip1_match_start) < min_gap for existing in candidates):
            continue
        if any(abs(existing.clip2_match_start_sec - clip2_match_start) < min_gap for existing in candidates):
            continue

        base_duration = float(config.similar_match_window_sec)
        export_start_1 = max(0.0, clip1_match_start - margin_before)
        export_start_2 = max(0.0, clip2_match_start - margin_before)
        export_duration = min(
            base_duration + margin_before + margin_after,
            duration1 - export_start_1,
            duration2 - export_start_2,
            m1.duration_sec - export_start_1,
            m2.duration_sec - export_start_2,
        )
        if export_duration <= 0.5:
            continue

        candidates.append(
            AlignSegment(
                segment_id=f"seg_{len(candidates) + 1:02d}",
                rank=len(candidates) + 1,
                is_best=False,
                clip1_match_start_sec=clip1_match_start,
                clip2_match_start_sec=clip2_match_start,
                match_duration_sec=base_duration,
                clip1_export_start_sec=export_start_1,
                clip2_export_start_sec=export_start_2,
                export_duration_sec=export_duration,
                score=score,
                onset_score=onset_score,
                chroma_score=chroma_score,
                tempogram_score=temp_score,
                note="综合匹配: onset/chroma/tempogram",
            )
        )
        if len(candidates) >= max_segments:
            break

    if not candidates:
        raise RuntimeError("未找到足够相似的音频片段，请尝试降低相似度阈值或缩短窗口")

    candidates[0].is_best = True
    best = candidates[0]
    warnings: list[str] = []
    if best.score < float(config.audio_confidence_floor):
        warnings.append(f"最佳片段置信度较低({best.score:.3f})，建议人工复核")

    return AlignResult(
        clip1_anchor_sec=float(best.clip1_match_start_sec),
        clip2_anchor_sec=float(best.clip2_match_start_sec),
        offset_sec=float(best.clip2_match_start_sec - best.clip1_match_start_sec),
        clip1_start_sec=float(best.clip1_export_start_sec),
        clip2_start_sec=float(best.clip2_export_start_sec),
        output_duration_sec=float(best.export_duration_sec),
        confidence=float(best.score),
        method="audio_similar_segments_local",
        warnings=warnings or None,
        segments=candidates,
        best_segment_index=0,
    )


def _prepare_audio_input(media_path: Path, target_path: Path) -> Path:
    suffix = media_path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        shutil.copy2(media_path, target_path)
        return target_path
    return extract_audio_track(media_path, target_path)


def _align_via_remote(a1: Path, a2: Path, config: AlignConfig) -> AlignResult:
    remote = OtogeAlignClient(config).align_audio(a1, a2, config)
    return AlignResult(
        clip1_anchor_sec=float(remote.anchor_a_sec),
        clip2_anchor_sec=float(remote.anchor_b_sec),
        offset_sec=float(remote.offset_sec),
        clip1_start_sec=float(remote.start_a_sec),
        clip2_start_sec=float(remote.start_b_sec),
        output_duration_sec=float(remote.overlap_duration_sec),
        confidence=float(remote.confidence),
        method=remote.method,
        warnings=remote.warnings,
    )


def _align_via_local(a1: Path, a2: Path, config: AlignConfig) -> AlignResult:
    try:
        from .audalign import AudioAlignParams, align_audio_pair
    except ImportError:  # pragma: no cover
        from audalign import AudioAlignParams, align_audio_pair

    params = AudioAlignParams(
        sample_rate=int(config.audio_sr),
        hop_length=int(config.audio_hop_length),
        n_fft=int(config.audio_n_fft),
        search_range_sec=float(config.audio_search_range_sec),
        min_overlap_sec=float(config.audio_min_overlap_sec),
        confidence_floor=float(config.audio_confidence_floor),
        max_duration_sec=float(config.audio_max_duration_sec),
    )
    local = align_audio_pair(params=params, audio_a_path=a1, audio_b_path=a2)
    return AlignResult(
        clip1_anchor_sec=float(local.anchor_a_sec),
        clip2_anchor_sec=float(local.anchor_b_sec),
        offset_sec=float(local.offset_sec),
        clip1_start_sec=float(local.start_a_sec),
        clip2_start_sec=float(local.start_b_sec),
        output_duration_sec=float(local.overlap_duration_sec),
        confidence=float(local.confidence),
        method="audio_local",
        warnings=list(local.warnings),
    )


def align_audio_media(clip1_path: Path, clip2_path: Path, config: AlignConfig) -> AlignResult:
    m1 = probe_media(clip1_path)
    m2 = probe_media(clip2_path)
    if not m1.has_audio:
        raise RuntimeError("Clip1 不包含音轨，无法进行音频对齐")
    if not m2.has_audio:
        raise RuntimeError("Clip2 不包含音轨，无法进行音频对齐")

    with tempfile.TemporaryDirectory(prefix="maimai-audalign-") as td:
        tmp = Path(td)
        a1 = _prepare_audio_input(clip1_path, tmp / "clip1.wav")
        a2 = _prepare_audio_input(clip2_path, tmp / "clip2.wav")

        if (config.align_mode or "standard").strip().lower() == "similar_segments":
            result = _select_similar_segments(a1, a2, m1, m2, config)
            output_duration = float(result.output_duration_sec)
            merged_warnings = list(result.warnings or [])
            return AlignResult(
                clip1_anchor_sec=float(result.clip1_anchor_sec),
                clip2_anchor_sec=float(result.clip2_anchor_sec),
                offset_sec=float(result.offset_sec),
                clip1_start_sec=float(result.clip1_start_sec),
                clip2_start_sec=float(result.clip2_start_sec),
                output_duration_sec=float(output_duration),
                confidence=float(result.confidence),
                method=result.method,
                warnings=merged_warnings or None,
                segments=result.segments,
                best_segment_index=result.best_segment_index,
            )

        backend = (config.align_backend or "remote").strip().lower()
        if backend not in {"remote", "local"}:
            backend = "remote"

        warnings: list[str] = []
        if backend == "local":
            result = _align_via_local(a1, a2, config)
        else:
            try:
                result = _align_via_remote(a1, a2, config)
            except Exception as exc:
                if not config.align_fallback_to_local:
                    raise
                result = _align_via_local(a1, a2, config)
                warnings.append(f"外部 API 对齐失败，已回退内置 audalign：{exc}")

    output_duration = min(
        m1.duration_sec - float(result.clip1_start_sec),
        m2.duration_sec - float(result.clip2_start_sec),
        float(result.output_duration_sec),
    )
    if output_duration <= 0.5:
        raise RuntimeError("音频对齐后无足够重叠时长，无法导出")

    merged_warnings = [*(result.warnings or []), *warnings]
    return AlignResult(
        clip1_anchor_sec=float(result.clip1_anchor_sec),
        clip2_anchor_sec=float(result.clip2_anchor_sec),
        offset_sec=float(result.offset_sec),
        clip1_start_sec=float(result.clip1_start_sec),
        clip2_start_sec=float(result.clip2_start_sec),
        output_duration_sec=float(output_duration),
        confidence=float(result.confidence),
        method=result.method,
        warnings=merged_warnings or None,
    )
