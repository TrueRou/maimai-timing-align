from __future__ import annotations

from maimai_timing_align.models import (
    AlignResult,
    AlignSegment,
    OsuBatchMatchResult,
    OsuBeatmapCandidate,
)


def test_osu_batch_result_keeps_candidate_segments() -> None:
    segment = AlignSegment(
        segment_id="seg_01",
        rank=1,
        is_best=True,
        clip1_match_start_sec=10.0,
        clip2_match_start_sec=20.0,
        match_duration_sec=8.0,
        clip1_export_start_sec=8.0,
        clip2_export_start_sec=18.0,
        export_duration_sec=12.0,
        score=0.91,
        onset_score=0.9,
        chroma_score=0.92,
        tempogram_score=0.91,
    )
    result = AlignResult(
        clip1_anchor_sec=10.0,
        clip2_anchor_sec=20.0,
        offset_sec=10.0,
        clip1_start_sec=8.0,
        clip2_start_sec=18.0,
        output_duration_sec=12.0,
        confidence=0.91,
        method="audio_similar_segments_local",
        segments=[segment],
        best_segment_index=0,
    )
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
    batch_item = OsuBatchMatchResult(candidate=candidate, audio_path=None, result=result)  # type: ignore[arg-type]
    assert batch_item.result.segments is not None
    assert batch_item.result.segments[0].rank == 1
    assert batch_item.result.segments[0].is_best is True
