from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .api import OtogeAlignClient
from .media import extract_audio_track, probe_media
from .models import AlignConfig, AlignResult

AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma"}


def _prepare_audio_input(media_path: Path, target_path: Path) -> Path:
    suffix = media_path.suffix.lower()
    if suffix in AUDIO_SUFFIXES:
        shutil.copy2(media_path, target_path)
        return target_path
    return extract_audio_track(media_path, target_path)

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
        remote = OtogeAlignClient(config).align_audio(a1, a2, config)

    output_duration = min(
        m1.duration_sec - float(remote.start_a_sec),
        m2.duration_sec - float(remote.start_b_sec),
        float(remote.overlap_duration_sec),
    )
    if output_duration <= 0.5:
        raise RuntimeError("音频对齐后无足够重叠时长，无法导出")

    return AlignResult(
        clip1_anchor_sec=float(remote.anchor_a_sec),
        clip2_anchor_sec=float(remote.anchor_b_sec),
        offset_sec=float(remote.offset_sec),
        clip1_start_sec=float(remote.start_a_sec),
        clip2_start_sec=float(remote.start_b_sec),
        output_duration_sec=float(output_duration),
        confidence=float(remote.confidence),
        method=remote.method,
        warnings=remote.warnings,
    )
