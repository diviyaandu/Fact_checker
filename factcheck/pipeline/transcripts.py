"""Transcript stage: cache lookup, fetching (official captions / Whisper), saving."""

from factcheck import cache
from factcheck.logging_utils import logger
from factcheck.transcript import get_official_transcript, transcribe_with_whisper


def get_cached_transcript(video_id: str, use_cache: bool) -> dict | None:
    """Returns {"segments":..., "method":...} from cache, or None on a miss
    (or when caching is disabled). Logs the cache hit/miss either way."""
    cached = cache.get("transcript", video_id) if use_cache else None

    if cached:
        logger.info("CACHE hit | namespace=transcript | video_id=%s", video_id)
        return cached

    logger.info("CACHE miss | namespace=transcript | video_id=%s", video_id)
    return None


def fetch_official_transcript(video_id: str) -> list[dict] | None:
    """Thin wrapper around transcript.get_official_transcript for symmetry/clarity."""
    return get_official_transcript(video_id)


def fetch_whisper_transcript(video_id: str) -> list[dict]:
    """Thin wrapper around transcript.transcribe_with_whisper. May raise."""
    return transcribe_with_whisper(video_id)


def save_transcript(video_id: str, segments: list[dict], method: str) -> None:
    logger.info(
        "TRANSCRIPT ready | video_id=%s | method=%s | segments=%d",
        video_id, method, len(segments),
    )
    cache.set("transcript", video_id, {"segments": segments, "method": method})
