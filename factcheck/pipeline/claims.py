"""Claims stage: cache lookup, extraction, saving."""

from factcheck import cache
from factcheck.claims import extract_claims
from factcheck.logging_utils import logger
from factcheck.models import Claim


def get_cached_claims(video_id: str, use_cache: bool) -> list[Claim] | None:
    """Returns cached claims for a video, or None on a miss/disabled cache."""
    cached = cache.get("claims", video_id) if use_cache else None

    if cached is not None:
        logger.info("CACHE hit | namespace=claims | video_id=%s", video_id)
        return [Claim(**c) for c in cached]

    logger.info("CACHE miss | namespace=claims | video_id=%s", video_id)
    return None


def fetch_claims(segments: list[dict]) -> list[Claim]:
    """Thin wrapper around claims.extract_claims (MAX_CLAIMS is enforced there)."""
    return extract_claims(segments)


def save_claims(video_id: str, claims: list[Claim]) -> None:
    cache.set("claims", video_id, [c.model_dump() for c in claims])
