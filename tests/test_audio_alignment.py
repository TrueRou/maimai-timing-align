from __future__ import annotations

from pathlib import Path

from maimai_timing_align.analysis import align_audio_media
from maimai_timing_align.models import AlignConfig, AlignResult


def test_alignment_router_uses_audio_mode_only(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_align(p1: Path, p2: Path, cfg: AlignConfig) -> AlignResult:
        captured["p1"] = p1
        captured["p2"] = p2
        captured["cfg"] = cfg
        return AlignResult(
            clip1_anchor_sec=1.0,
            clip2_anchor_sec=2.0,
            offset_sec=1.0,
            clip1_start_sec=1.0,
            clip2_start_sec=2.0,
            output_duration_sec=10.0,
            confidence=0.88,
        )

    monkeypatch.setattr("maimai_timing_align.alignment_router.align_audio_media", _fake_align)

    cfg = AlignConfig()
    out = align_audio_media(Path("a.mp4"), Path("b.mp3"), cfg)
    assert isinstance(captured.get("cfg"), AlignConfig)
    assert out.confidence == 0.88
