from __future__ import annotations

import shutil
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import httpx
from ossapi import Ossapi
from ossapi.enums import BeatmapsetSearchCategory, BeatmapsetSearchMode, BeatmapsetSearchSort

try:
    from .models import AlignConfig, OsuBeatmapCandidate
except ImportError:  # pragma: no cover
    from models import AlignConfig, OsuBeatmapCandidate

OSU_DIRECT_DOWNLOAD = "https://osu.direct/api/d/{beatmapset_id}"
OSU_AUDIO_EXTENSIONS = (".mp3",)


def _strict_bpm_equal(lhs: float | None, rhs: float) -> bool:
    if lhs is None:
        return False
    return float(lhs).is_integer() and float(rhs).is_integer() and int(lhs) == int(rhs) or abs(float(lhs) - float(rhs)) < 1e-9


def _build_query(config: AlignConfig) -> str | None:
    parts: list[str] = []
    if config.osu_bpm > 0:
        parts.append(f"bpm={config.osu_bpm}")
    if config.osu_query.strip():
        parts.append(config.osu_query.strip())
    if config.osu_artist.strip():
        parts.append(f"artist={config.osu_artist.strip()}")
    if config.osu_creator.strip():
        parts.append(f"creator={config.osu_creator.strip()}")
    if config.osu_version.strip():
        parts.append(f"version={config.osu_version.strip()}")
    return " ".join(parts) or None


def search_osu_candidates(config: AlignConfig) -> list[OsuBeatmapCandidate]:
    if not config.osu_client_id.strip() or not config.osu_client_secret.strip():
        raise RuntimeError("请先填写 osu! OAuth Client ID 和 Client Secret")
    if config.osu_bpm <= 0:
        raise RuntimeError("osu BPM 必须大于 0")

    api = Ossapi(int(config.osu_client_id), str(config.osu_client_secret))
    mode = {
        "any": BeatmapsetSearchMode.ANY,
        "osu": BeatmapsetSearchMode.OSU,
        "taiko": BeatmapsetSearchMode.TAIKO,
        "fruits": BeatmapsetSearchMode.CATCH,
        "mania": BeatmapsetSearchMode.MANIA,
    }[config.osu_mode]
    category = {
        "has_leaderboard": BeatmapsetSearchCategory.HAS_LEADERBOARD,
        "ranked": BeatmapsetSearchCategory.RANKED,
        "loved": BeatmapsetSearchCategory.LOVED,
        "qualified": BeatmapsetSearchCategory.QUALIFIED,
        "pending": BeatmapsetSearchCategory.PENDING,
        "graveyard": BeatmapsetSearchCategory.GRAVEYARD,
    }[config.osu_category]

    res = api.search_beatmapsets(
        query=_build_query(config),
        mode=mode,
        category=category,
        sort=BeatmapsetSearchSort.PLAYS_DESCENDING,
    )

    out: list[OsuBeatmapCandidate] = []
    seen_beatmapsets: set[int] = set()
    for beatmapset in res.beatmapsets:
        if int(beatmapset.id) in seen_beatmapsets:
            continue
        beatmaps = list(beatmapset.beatmaps or [])
        for beatmap in beatmaps:
            if not _strict_bpm_equal(getattr(beatmap, "bpm", None), float(config.osu_bpm)):
                continue
            seen_beatmapsets.add(int(beatmapset.id))
            out.append(
                OsuBeatmapCandidate(
                    beatmap_id=int(beatmap.id),
                    beatmapset_id=int(beatmapset.id),
                    artist=str(beatmapset.artist),
                    title=str(beatmapset.title),
                    version=str(beatmap.version),
                    creator=str(beatmapset.creator),
                    bpm=float(beatmap.bpm or beatmapset.bpm or 0.0),
                    mode=str(beatmap.mode),
                    source_url=str(getattr(beatmap, "url", "") or ""),
                )
            )
            if len(out) >= int(config.osu_batch_limit):
                return out
            break
    return out


def download_osz_and_extract_first_mp3(candidate: OsuBeatmapCandidate, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    osz_path = target_dir / f"osu_{candidate.beatmapset_id}.osz"
    extract_dir = target_dir / f"osu_{candidate.beatmapset_id}"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with httpx.stream("GET", OSU_DIRECT_DOWNLOAD.format(beatmapset_id=candidate.beatmapset_id), timeout=60.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        with osz_path.open("wb") as fw:
            for chunk in resp.iter_bytes():
                if chunk:
                    fw.write(chunk)

    zip_path = osz_path.with_suffix(".zip")
    osz_path.replace(zip_path)
    try:
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except BadZipFile as exc:
        raise RuntimeError(f"beatmapset {candidate.beatmapset_id} 下载文件损坏，无法解压") from exc
    except Exception as exc:
        raise RuntimeError(f"beatmapset {candidate.beatmapset_id} 解压失败：{exc}") from exc

    mp3_files = sorted(
        p for p in extract_dir.rglob("*") if p.is_file() and p.suffix.lower() in OSU_AUDIO_EXTENSIONS
    )
    if not mp3_files:
        raise RuntimeError(f"beatmapset {candidate.beatmapset_id} 解压后未找到 mp3 文件")
    return mp3_files[0]
