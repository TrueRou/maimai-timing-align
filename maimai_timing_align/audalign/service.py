from __future__ import annotations

import tempfile
from pathlib import Path

from .core import estimate_offset_from_onset_vectors
from .features import extract_onset_envelope, load_audio_mono
from .models import AudioAlignDiagnostics, AudioAlignParams, AudioAlignResult


def _write_upload_to_temp(data: bytes, suffix: str) -> Path:
    suffix = suffix if suffix.startswith(".") else f".{suffix}" if suffix else ".bin"
    tf = tempfile.NamedTemporaryFile(prefix="audalign-", suffix=suffix, delete=False)
    tf.write(data)
    tf.flush()
    tf.close()
    return Path(tf.name)


def align_audio_pair(
    params: AudioAlignParams,
    audio_a_path: Path | None = None,
    audio_b_path: Path | None = None,
    audio_a_bytes: bytes | None = None,
    audio_b_bytes: bytes | None = None,
    audio_a_suffix: str = ".bin",
    audio_b_suffix: str = ".bin",
) -> AudioAlignResult:
    temp_paths: list[Path] = []
    try:
        if audio_a_bytes is not None:
            audio_a_path = _write_upload_to_temp(audio_a_bytes, audio_a_suffix)
            temp_paths.append(audio_a_path)
        if audio_b_bytes is not None:
            audio_b_path = _write_upload_to_temp(audio_b_bytes, audio_b_suffix)
            temp_paths.append(audio_b_path)

        if audio_a_path is None or audio_b_path is None:
            raise RuntimeError("必须同时提供两段音频（文件路径或上传文件）")

        y_a, sr_a = load_audio_mono(audio_a_path, params)
        y_b, sr_b = load_audio_mono(audio_b_path, params)
        sr = min(sr_a, sr_b)

        onset_a = extract_onset_envelope(y_a, sr, params)
        onset_b = extract_onset_envelope(y_b, sr, params)

        hop_sec = float(params.hop_length) / float(sr)
        effective_hop_sec = hop_sec

        offset, confidence = estimate_offset_from_onset_vectors(
            onset_a,
            onset_b,
            hop_sec=effective_hop_sec,
            search_range_sec=float(params.search_range_sec),
            min_overlap_sec=float(params.min_overlap_sec),
        )

        start_a = 0.0
        start_b = start_a + offset
        if start_b < 0:
            start_a -= start_b
            start_b = 0.0

        dur_a = float(y_a.size) / float(sr_a)
        dur_b = float(y_b.size) / float(sr_b)
        overlap = min(dur_a - start_a, dur_b - start_b)
        if overlap <= 0.5:
            raise RuntimeError("对齐后无足够重叠时长")

        warnings: list[str] = []
        if confidence < float(params.confidence_floor):
            warnings.append(f"对齐置信度较低({confidence:.3f} < {params.confidence_floor:.3f})，建议人工复核")

        return AudioAlignResult(
            anchor_a_sec=start_a,
            anchor_b_sec=start_b,
            offset_sec=offset,
            start_a_sec=start_a,
            start_b_sec=start_b,
            overlap_duration_sec=overlap,
            confidence=confidence,
            method=params.method,
            warnings=warnings,
            diagnostics=AudioAlignDiagnostics(
                hop_sec_effective=effective_hop_sec,
                onset_frames_a=int(onset_a.size),
                onset_frames_b=int(onset_b.size),
            ),
        )
    finally:
        for p in temp_paths:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
