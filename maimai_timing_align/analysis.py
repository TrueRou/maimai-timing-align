from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np

try:
    from .api import OtogeAlignClient
    from .media import extract_audio_track, probe_media
    from .models import AlignConfig, AlignResult, AlignSegment, OsuBatchMatchResult
    from .osu import download_osz_and_extract_first_mp3, search_osu_candidates
except ImportError:  # pragma: no cover
    from api import OtogeAlignClient
    from media import extract_audio_track, probe_media
    from models import AlignConfig, AlignResult, AlignSegment, OsuBatchMatchResult
    from osu import download_osz_and_extract_first_mp3, search_osu_candidates

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


def _score_candidate_pair(
    start_a: int,
    start_b: int,
    features1: dict[str, np.ndarray],
    features2: dict[str, np.ndarray],
    window_frames: int,
    hop_sec: float,
    config: AlignConfig,
) -> tuple[float, float, float, float, float, float] | None:
    score, onset_score, chroma_score, temp_score = _weighted_similarity(
        features1,
        features2,
        start_a,
        start_b,
        window_frames,
        config,
    )
    if score < float(config.similar_similarity_floor):
        return None
    return (
        score,
        onset_score,
        chroma_score,
        temp_score,
        float(start_a) * hop_sec,
        float(start_b) * hop_sec,
    )


def _similar_num_workers(config: AlignConfig) -> int:
    value = getattr(config, "similar_num_workers", 0)
    try:
        workers = int(value)
    except (TypeError, ValueError):
        workers = 0
    return workers


def _refine_segment_bounds(
    feature_a: dict[str, np.ndarray],
    feature_b: dict[str, np.ndarray],
    start_a: int,
    start_b: int,
    seed_length_frames: int,
    frames1: int,
    frames2: int,
    step_frames: int,
    config: AlignConfig,
) -> tuple[int, int, int, float, float, float, float]:
    best_start_a = start_a
    best_start_b = start_b
    best_length = seed_length_frames
    best_score, best_onset, best_chroma, best_temp = _weighted_similarity(
        feature_a,
        feature_b,
        start_a,
        start_b,
        seed_length_frames,
        config,
    )

    expand_floor = max(
        float(config.similar_similarity_floor),
        best_score - 0.06,
    )

    current_start_a = start_a
    current_start_b = start_b
    current_length = seed_length_frames

    def _is_better(score: float, length: int, start1: int, start2: int) -> bool:
        if score > best_score + 1e-9:
            return True
        if abs(score - best_score) <= 1e-9 and length > best_length:
            return True
        if (
            abs(score - best_score) <= 1e-9
            and length == best_length
            and (start1 < best_start_a or start2 < best_start_b)
        ):
            return True
        return False

    while True:
        expanded = False

        if current_start_a - step_frames >= 0 and current_start_b - step_frames >= 0:
            cand_start_a = current_start_a - step_frames
            cand_start_b = current_start_b - step_frames
            cand_length = current_length + step_frames
            score, onset_score, chroma_score, temp_score = _weighted_similarity(
                feature_a,
                feature_b,
                cand_start_a,
                cand_start_b,
                cand_length,
                config,
            )
            if score >= expand_floor:
                current_start_a = cand_start_a
                current_start_b = cand_start_b
                current_length = cand_length
                expanded = True
                if _is_better(score, cand_length, cand_start_a, cand_start_b):
                    best_start_a = cand_start_a
                    best_start_b = cand_start_b
                    best_length = cand_length
                    best_score = score
                    best_onset = onset_score
                    best_chroma = chroma_score
                    best_temp = temp_score

        if current_start_a + current_length + step_frames <= frames1 and current_start_b + current_length + step_frames <= frames2:
            cand_start_a = current_start_a
            cand_start_b = current_start_b
            cand_length = current_length + step_frames
            score, onset_score, chroma_score, temp_score = _weighted_similarity(
                feature_a,
                feature_b,
                cand_start_a,
                cand_start_b,
                cand_length,
                config,
            )
            if score >= expand_floor:
                current_length = cand_length
                expanded = True
                if _is_better(score, cand_length, cand_start_a, cand_start_b):
                    best_start_a = cand_start_a
                    best_start_b = cand_start_b
                    best_length = cand_length
                    best_score = score
                    best_onset = onset_score
                    best_chroma = chroma_score
                    best_temp = temp_score

        if not expanded:
            break

    return best_start_a, best_start_b, best_length, best_score, best_onset, best_chroma, best_temp


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
    min_gap = max(0.0, float(config.similar_min_segment_gap_sec))
    margin_before = max(0.0, float(config.similar_margin_before_sec))
    margin_after = max(0.0, float(config.similar_margin_after_sec))
    max_segments = max(1, int(config.similar_max_segments))

    candidate_rows: list[tuple[float, float, float, float, float, float]] = []
    tasks = [
        (start_a, start_b)
        for start_a in range(0, frames1 - window_frames + 1, step_frames)
        for start_b in range(0, frames2 - window_frames + 1, step_frames)
    ]
    configured_workers = _similar_num_workers(config)
    max_workers = configured_workers if configured_workers > 0 else min(32, (os.cpu_count() or 1))

    if max_workers <= 1 or len(tasks) <= 1:
        for start_a, start_b in tasks:
            scored = _score_candidate_pair(start_a, start_b, features1, features2, window_frames, hop_sec, config)
            if scored is not None:
                candidate_rows.append(scored)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _score_candidate_pair,
                    start_a,
                    start_b,
                    features1,
                    features2,
                    window_frames,
                    hop_sec,
                    config,
                )
                for start_a, start_b in tasks
            ]
            for future in futures:
                scored = future.result()
                if scored is not None:
                    candidate_rows.append(scored)

    candidate_rows.sort(key=lambda item: item[0], reverse=True)
    for score, onset_score, chroma_score, temp_score, clip1_match_start, clip2_match_start in candidate_rows:
        if any(abs(existing.clip1_match_start_sec - clip1_match_start) < min_gap for existing in candidates):
            continue
        if any(abs(existing.clip2_match_start_sec - clip2_match_start) < min_gap for existing in candidates):
            continue

        start_a_frames = int(round(clip1_match_start / max(hop_sec, 1e-6)))
        start_b_frames = int(round(clip2_match_start / max(hop_sec, 1e-6)))
        (
            refined_start_a,
            refined_start_b,
            refined_length_frames,
            score,
            onset_score,
            chroma_score,
            temp_score,
        ) = _refine_segment_bounds(
            features1,
            features2,
            start_a_frames,
            start_b_frames,
            window_frames,
            frames1,
            frames2,
            step_frames,
            config,
        )

        clip1_match_start = float(refined_start_a) * hop_sec
        clip2_match_start = float(refined_start_b) * hop_sec
        base_duration = float(refined_length_frames) * hop_sec
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
                note="综合匹配: onset/chroma/tempogram，长度自动扩展",
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


def batch_match_osu_candidates(clip1_path: Path, work_dir: Path, config: AlignConfig) -> list[OsuBatchMatchResult]:
    candidates = search_osu_candidates(config)
    if not candidates:
        raise RuntimeError("未找到符合 BPM 和筛选条件的 osu 候选谱面")

    out: list[OsuBatchMatchResult] = []
    errors: list[str] = []
    cached_audio_paths: dict[int, Path] = {}
    cached_results: dict[int, AlignResult] = {}
    for idx, candidate in enumerate(candidates, start=1):
        candidate_dir = work_dir / f"osu_candidate_{idx:02d}_{candidate.beatmapset_id}"
        try:
            beatmapset_id = int(candidate.beatmapset_id)
            audio_path = cached_audio_paths.get(beatmapset_id)
            if audio_path is None:
                audio_path = download_osz_and_extract_first_mp3(candidate, candidate_dir)
                cached_audio_paths[beatmapset_id] = audio_path

            result = cached_results.get(beatmapset_id)
            if result is None:
                result = align_audio_media(clip1_path, audio_path, config)
                cached_results[beatmapset_id] = result

            out.append(OsuBatchMatchResult(candidate=candidate, audio_path=audio_path, result=result))
        except Exception as exc:
            errors.append(f"{candidate.artist} - {candidate.title} [{candidate.version}]：{exc}")

    if not out:
        raise RuntimeError("所有 osu 候选匹配均失败：\n" + "\n".join(errors))

    out.sort(key=lambda item: float(item.result.confidence), reverse=True)
    if errors and out:
        top = out[0]
        merged = [*(top.result.warnings or []), *errors]
        top.result = AlignResult(
            clip1_anchor_sec=top.result.clip1_anchor_sec,
            clip2_anchor_sec=top.result.clip2_anchor_sec,
            offset_sec=top.result.offset_sec,
            clip1_start_sec=top.result.clip1_start_sec,
            clip2_start_sec=top.result.clip2_start_sec,
            output_duration_sec=top.result.output_duration_sec,
            confidence=top.result.confidence,
            method=top.result.method,
            warnings=merged,
            segments=top.result.segments,
            best_segment_index=top.result.best_segment_index,
        )
    return out
