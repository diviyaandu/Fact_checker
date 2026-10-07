"""Read-only questions about what is already cached."""

from factcheck import cache


def claim_cache_key(claim_text: str) -> str:
    """Cache key for one claim's sources + verdict (shared by lookup and writes)."""
    return cache.make_key("claim", claim_text)


def is_fully_cached(video_id: str, use_cache: bool) -> bool:
    """
    True when analysing this video would need NO external calls — the transcript,
    the extracted claims and every claim's sources + verdict are all in the cache
    (so no YouTube/Whisper, Groq or Tavily request would be made).

    Read-only and silent (no hit/miss logging), so the UI can call it freely to
    decide whether the Analyze cooldown applies. Mirrors the checks made by
    get_cached_transcript / get_cached_claims / get_claim_verdicts.
    """
    if not use_cache:
        return False

    if not cache.get("transcript", video_id):
        return False

    cached_claims = cache.get("claims", video_id)
    if cached_claims is None:
        return False

    return all(
        cache.get("claim_results", claim_cache_key(c["claim_text"]))
        for c in cached_claims
    )
