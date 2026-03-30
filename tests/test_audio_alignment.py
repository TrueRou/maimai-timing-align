from __future__ import annotations

from pathlib import Path

from maimai_timing_align.analysis import align_audio_media
from maimai_timing_align.models import AlignConfig, AlignResult, AlignSegment


class _MediaInfo:
    def __init__(self, duration_sec: float, has_audio: bool = True, has_video: bool = True):
        self.duration_sec = duration_sec
        self.has_audio = has_audio
        self.has_video = has_video


def test_align_audio_media_uses_remote_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "maimai_timing_align.analysis.probe_media",
        lambda _path: _MediaInfo(duration_sec=120.0),
    )
    monkeypatch.setattr(
        "maimai_timing_align.analysis._prepare_audio_input",
        lambda media_path, _target_path: media_path,
    )

    def _fake_remote(a1: Path, a2: Path, cfg: AlignConfig) -> AlignResult:
        assert a1 == Path("clip1.mp4")
        assert a2 == Path("clip2.mp3")
        assert cfg.align_backend == "remote"
        return AlignResult(
            clip1_anchor_sec=1.0,
            clip2_anchor_sec=2.0,
            offset_sec=1.0,
            clip1_start_sec=1.0,
            clip2_start_sec=2.0,
            output_duration_sec=25.0,
            confidence=0.91,
            method="audio_remote",
            warnings=["remote warning"],
        )

    monkeypatch.setattr("maimai_timing_align.analysis._align_via_remote", _fake_remote)

    cfg = AlignConfig(align_backend="remote", align_fallback_to_local=True)
    out = align_audio_media(Path("clip1.mp4"), Path("clip2.mp3"), cfg)
    assert out.method == "audio_remote"
    assert out.output_duration_sec == 25.0
    assert out.warnings == ["remote warning"]


def test_align_audio_media_uses_local_backend(monkeypatch) -> None:
    monkeypatch.setattr(
        "maimai_timing_align.analysis.probe_media",
        lambda _path: _MediaInfo(duration_sec=90.0),
    )
    monkeypatch.setattr(
        "maimai_timing_align.analysis._prepare_audio_input",
        lambda media_path, _target_path: media_path,
    )

    def _fake_local(a1: Path, a2: Path, cfg: AlignConfig) -> AlignResult:
        assert a1 == Path("clip1.mp4")
        assert a2 == Path("clip2.mp3")
        assert cfg.align_backend == "local"
        return AlignResult(
            clip1_anchor_sec=0.0,
            clip2_anchor_sec=0.5,
            offset_sec=0.5,
            clip1_start_sec=0.0,
            clip2_start_sec=0.5,
            output_duration_sec=30.0,
            confidence=0.72,
            method="audio_local",
            warnings=[],
        )

    monkeypatch.setattr("maimai_timing_align.analysis._align_via_local", _fake_local)

    cfg = AlignConfig(align_backend="local", align_fallback_to_local=False)
    out = align_audio_media(Path("clip1.mp4"), Path("clip2.mp3"), cfg)
    assert out.method == "audio_local"
    assert out.output_duration_sec == 30.0
    assert out.warnings is None


def test_align_audio_media_falls_back_to_local(monkeypatch) -> None:
    monkeypatch.setattr(
        "maimai_timing_align.analysis.probe_media",
        lambda _path: _MediaInfo(duration_sec=80.0),
    )
    monkeypatch.setattr(
        "maimai_timing_align.analysis._prepare_audio_input",
        lambda media_path, _target_path: media_path,
    )

    def _fail_remote(_a1: Path, _a2: Path, _cfg: AlignConfig) -> AlignResult:
        raise RuntimeError("remote unavailable")

    def _fake_local(_a1: Path, _a2: Path, _cfg: AlignConfig) -> AlignResult:
        return AlignResult(
            clip1_anchor_sec=0.0,
            clip2_anchor_sec=1.0,
            offset_sec=1.0,
            clip1_start_sec=0.0,
            clip2_start_sec=1.0,
            output_duration_sec=18.0,
            confidence=0.66,
            method="audio_local",
            warnings=["local warning"],
        )

    monkeypatch.setattr("maimai_timing_align.analysis._align_via_remote", _fail_remote)
    monkeypatch.setattr("maimai_timing_align.analysis._align_via_local", _fake_local)

    cfg = AlignConfig(align_backend="remote", align_fallback_to_local=True)
    out = align_audio_media(Path("clip1.mp4"), Path("clip2.mp3"), cfg)
    assert out.method == "audio_local"
    assert out.warnings is not None
    assert any("回退内置 audalign" in warning for warning in out.warnings)
    assert any("local warning" in warning for warning in out.warnings)


def test_align_audio_media_uses_similar_segments_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        "maimai_timing_align.analysis.probe_media",
        lambda _path: _MediaInfo(duration_sec=140.0),
    )
    monkeypatch.setattr(
        "maimai_timing_align.analysis._prepare_audio_input",
        lambda media_path, _target_path: media_path,
    )

    def _fake_similar(a1: Path, a2: Path, m1, m2, cfg: AlignConfig) -> AlignResult:
        assert a1 == Path("clip1.mp4")
        assert a2 == Path("clip2.mp3")
        assert cfg.align_mode == "similar_segments"
        assert m1.duration_sec == 140.0
        assert m2.duration_sec == 140.0
        segment = AlignSegment(
            segment_id="seg_01",
            rank=1,
            is_best=True,
            clip1_match_start_sec=32.0,
            clip2_match_start_sec=48.0,
            match_duration_sec=12.0,
            clip1_export_start_sec=30.5,
            clip2_export_start_sec=46.5,
            export_duration_sec=15.5,
            score=0.81,
            onset_score=0.83,
            chroma_score=0.78,
            tempogram_score=0.82,
        )
        return AlignResult(
            clip1_anchor_sec=32.0,
            clip2_anchor_sec=48.0,
            offset_sec=16.0,
            clip1_start_sec=30.5,
            clip2_start_sec=46.5,
            output_duration_sec=15.5,
            confidence=0.81,
            method="audio_similar_segments_local",
            warnings=None,
            segments=[segment],
            best_segment_index=0,
        )

    monkeypatch.setattr("maimai_timing_align.analysis._select_similar_segments", _fake_similar)

    cfg = AlignConfig(align_mode="similar_segments", align_backend="local")
    out = align_audio_media(Path("clip1.mp4"), Path("clip2.mp3"), cfg)
    assert out.method == "audio_similar_segments_local"
    assert out.best_segment is not None
    assert out.best_segment.is_best is True
    assert out.best_segment.rank == 1
    assert out.output_duration_sec == 15.5
