from __future__ import annotations

from zipfile import ZipFile

from maimai_timing_align.exporter import result_from_segment_to_end
from maimai_timing_align.models import AlignSegment
from maimai_timing_align.osu import (
    _strict_bpm_equal,
    download_osz_and_extract_first_mp3,
    search_osu_candidates,
)


def test_strict_bpm_equal() -> None:
    assert _strict_bpm_equal(180.0, 180.0) is True
    assert _strict_bpm_equal(180, 180.0) is True
    assert _strict_bpm_equal(180.5, 180.0) is False
    assert _strict_bpm_equal(None, 180.0) is False


def test_download_osz_extracts_first_mp3(tmp_path, monkeypatch) -> None:
    zip_path = tmp_path / "payload.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr("audio_b.mp3", b"b")
        zf.writestr("audio_a.mp3", b"a")

    class _Resp:
        def __init__(self, payload: bytes):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self):
            yield self.payload

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    payload = zip_path.read_bytes()
    monkeypatch.setattr("maimai_timing_align.osu.httpx.stream", lambda *args, **kwargs: _Resp(payload))

    from maimai_timing_align.models import OsuBeatmapCandidate

    candidate = OsuBeatmapCandidate(
        beatmap_id=1,
        beatmapset_id=2,
        artist="artist",
        title="title",
        version="hard",
        creator="mapper",
        bpm=180.0,
        mode="osu",
        source_url="",
    )
    out = download_osz_and_extract_first_mp3(candidate, tmp_path / "extract")
    assert out.name == "audio_a.mp3"
    assert out.exists()


def test_download_osz_reports_bad_zip(tmp_path, monkeypatch) -> None:
    class _Resp:
        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self):
            yield b"not-a-zip"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("maimai_timing_align.osu.httpx.stream", lambda *args, **kwargs: _Resp())

    from maimai_timing_align.models import OsuBeatmapCandidate

    candidate = OsuBeatmapCandidate(
        beatmap_id=1,
        beatmapset_id=2,
        artist="artist",
        title="title",
        version="hard",
        creator="mapper",
        bpm=180.0,
        mode="osu",
        source_url="",
    )

    try:
        download_osz_and_extract_first_mp3(candidate, tmp_path / "extract_bad")
    except RuntimeError as exc:
        assert "下载文件损坏" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_result_from_segment_to_end_still_works() -> None:
    segment = AlignSegment(
        segment_id="seg_tail",
        rank=1,
        is_best=True,
        clip1_match_start_sec=10.0,
        clip2_match_start_sec=30.0,
        match_duration_sec=8.0,
        clip1_export_start_sec=8.0,
        clip2_export_start_sec=28.0,
        export_duration_sec=12.0,
        score=0.88,
        onset_score=0.9,
        chroma_score=0.86,
        tempogram_score=0.87,
    )
    result = result_from_segment_to_end(segment, clip1_duration_sec=120.0, clip2_duration_sec=70.0)
    assert result.output_duration_sec == 42.0


def test_search_osu_candidates_deduplicates_same_beatmapset(monkeypatch) -> None:
    class _Beatmap:
        def __init__(self, beatmap_id: int, bpm: float, version: str):
            self.id = beatmap_id
            self.bpm = bpm
            self.version = version
            self.mode = "osu"
            self.url = f"https://osu.ppy.sh/beatmaps/{beatmap_id}"

    class _Beatmapset:
        def __init__(self):
            self.id = 100
            self.artist = "artist"
            self.title = "title"
            self.creator = "mapper"
            self.bpm = 180.0
            self.beatmaps = [
                _Beatmap(1, 180.0, "Normal"),
                _Beatmap(2, 180.0, "Hard"),
            ]

    class _SearchResult:
        def __init__(self):
            self.beatmapsets = [_Beatmapset()]

    class _Api:
        def search_beatmapsets(self, **kwargs):
            return _SearchResult()

    monkeypatch.setattr("maimai_timing_align.osu.Ossapi", lambda *args, **kwargs: _Api())

    from maimai_timing_align.models import AlignConfig

    cfg = AlignConfig(
        osu_client_id="1",
        osu_client_secret="secret",
        osu_bpm=180.0,
        osu_batch_limit=5,
    )
    candidates = search_osu_candidates(cfg)
    assert len(candidates) == 1
    assert candidates[0].beatmapset_id == 100
