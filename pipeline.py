"""
pipeline.py

All non-UI business logic for the fact-checking pipeline: transcript
fetching, claim extraction, and evidence + verdict orchestration. app.py
should only call into these functions and handle Streamlit rendering.

Verdict generation batches claims that need an LLM call (see
fact_checker.fact_check_claims_batch) BATCH_SIZE at a time. Evidence search
for every uncached claim happens first, so the batches only ever contain
claims that already have their sources ready.
"""

import math

import cache_utils
from logging_utils import logger

from youtube import get_official_transcript
from transcription import transcribe_with_whisper
from claims import extract_claims, Claim
from search import search_evidence_for_claim, Source
from fact_checker import (
    fact_check_no_sources,
    fact_check_claims_batch,
    Verdict,
    BATCH_SIZE,
)


# ============================================================
# TRANSCRIPT
# ============================================================

def get_cached_transcript(video_id: str, use_cache: bool) -> dict | None:
    """Returns {"segments":..., "method":...} from cache, or None on a miss
    (or when caching is disabled). Logs the cache hit/miss either way."""
    cached = cache_utils.get("transcript", video_id) if use_cache else None

    if cached:
        logger.info("CACHE hit | namespace=transcript | video_id=%s", video_id)
        return cached

    logger.info("CACHE miss | namespace=transcript | video_id=%s", video_id)
    return None


def fetch_official_transcript(video_id: str) -> list[dict] | None:
    """Thin wrapper around youtube.get_official_transcript for symmetry/clarity."""
    return get_official_transcript(video_id)


def fetch_whisper_transcript(video_id: str) -> list[dict]:
    """Thin wrapper around transcription.transcribe_with_whisper. May raise."""
    return transcribe_with_whisper(video_id)


def save_transcript(video_id: str, segments: list[dict], method: str) -> None:
    logger.info(
        "TRANSCRIPT ready | video_id=%s | method=%s | segments=%d",
        video_id, method, len(segments),
    )
    cache_utils.set("transcript", video_id, {"segments": segments, "method": method})


# ============================================================
# CLAIMS
# ============================================================

def get_cached_claims(video_id: str, use_cache: bool) -> list[Claim] | None:
    """Returns cached claims for a video, or None on a miss/disabled cache."""
    cached = cache_utils.get("claims", video_id) if use_cache else None

    if cached is not None:
        logger.info("CACHE hit | namespace=claims | video_id=%s", video_id)
        return [Claim(**c) for c in cached]

    logger.info("CACHE miss | namespace=claims | video_id=%s", video_id)
    return None


def fetch_claims(segments: list[dict]) -> list[Claim]:
    """Thin wrapper around claims.extract_claims. MAX_CLAIMS=5 is enforced
    inside claims.py and is unchanged by this refactor."""
    return extract_claims(segments)


def save_claims(video_id: str, claims: list[Claim]) -> None:
    cache_utils.set("claims", video_id, [c.model_dump() for c in claims])


# ============================================================
# EVIDENCE + VERDICTS (batched)
# ============================================================

def get_claim_verdicts(claims: list[Claim], use_cache: bool) -> dict[int, dict]:
    """
    Resolves sources + verdict for every claim.

    Returns: {claim_id: {"sources": [Source], "search_meta": dict,
                          "verdict": Verdict, "from_cache": bool,
                          "search_error": str | None}}

    Flow:
      1. Check the per-claim cache first; cached claims need no further work.
      2. For every uncached claim, run evidence search (Tavily via search.py)
         up front, for ALL of them, before any LLM call is made.
      3. Claims that came back with zero sources are resolved locally
         (no LLM call needed) via fact_checker.fact_check_no_sources.
      4. Remaining claims (those with sources) are sent to the LLM in fixed
         batches of BATCH_SIZE via fact_checker.fact_check_claims_batch. The
         final batch may contain fewer than BATCH_SIZE claims (as few as 1) —
         this falls out naturally from the slicing below, no special case
         needed.
    """
    results: dict[int, dict] = {}
    pending: list[tuple[Claim, str]] = []

    # --- Step 1: cache lookup ---
    for claim in claims:
        claim_key = cache_utils.make_key("claim", claim.claim_text)
        cached_result = cache_utils.get("claim_results", claim_key) if use_cache else None

        if cached_result:
            logger.info("CACHE hit | namespace=claim_results | claim_id=%d", claim.claim_id)
            results[claim.claim_id] = {
                "sources": [Source(**s) for s in cached_result["sources"]],
                "search_meta": cached_result["search_meta"],
                "verdict": Verdict(**cached_result["verdict"]),
                "from_cache": True,
                "search_error": None,
            }
        else:
            logger.info("CACHE miss | namespace=claim_results | claim_id=%d", claim.claim_id)
            pending.append((claim, claim_key))

    if not pending:
        return results

    # --- Step 2: gather evidence for EVERY uncached claim first ---
    evidence: dict[int, tuple[list[Source], dict, str | None]] = {}

    for claim, _claim_key in pending:
        try:
            found_sources, search_meta = search_evidence_for_claim(
                claim.claim_text, claim.category, claim.search_queries
            )
            search_error = None
        except Exception as e:
            logger.exception("SEARCH failed | claim_id=%d", claim.claim_id)
            found_sources, search_meta = [], {}
            search_error = str(e)

        logger.info(
            "SEARCH complete | claim_id=%d | queries=%d | succeeded=%d | "
            "failed=%d | raw_results=%d | sources=%d",
            claim.claim_id,
            len(search_meta.get("queries_used", [])),
            search_meta.get("queries_succeeded", 0),
            search_meta.get("queries_failed", 0),
            search_meta.get("total_raw_results", 0),
            len(found_sources),
        )

        evidence[claim.claim_id] = (found_sources, search_meta, search_error)

    # --- Step 3: split into "resolve locally" vs "needs an LLM verdict" ---
    needs_llm: list[tuple[Claim, str, list[Source], dict]] = []

    for claim, claim_key in pending:
        found_sources, search_meta, search_error = evidence[claim.claim_id]

        if found_sources:
            needs_llm.append((claim, claim_key, found_sources, search_meta))
        else:
            verdict = fact_check_no_sources(search_meta)
            _record_and_cache(results, claim, claim_key, found_sources, search_meta, verdict, search_error)

    # --- Step 4: batch the LLM calls, BATCH_SIZE claims per Groq call ---
    total_batches = math.ceil(len(needs_llm) / BATCH_SIZE) if needs_llm else 0

    for batch_num in range(total_batches):
        batch = needs_llm[batch_num * BATCH_SIZE: (batch_num + 1) * BATCH_SIZE]
        claim_ids = [c.claim_id for c, _, _, _ in batch]

        logger.info(
            "BATCH start | batch=%d/%d | size=%d | claim_ids=%s",
            batch_num + 1, total_batches, len(batch), claim_ids,
        )

        items = [(claim.claim_text, sources) for claim, _, sources, _ in batch]
        verdicts = fact_check_claims_batch(items)

        logger.info(
            "BATCH complete | batch=%d/%d | size=%d | claim_ids=%s",
            batch_num + 1, total_batches, len(batch), claim_ids,
        )

        for (claim, claim_key, sources, search_meta), verdict in zip(batch, verdicts):
            _record_and_cache(results, claim, claim_key, sources, search_meta, verdict, None)

    return results


def _record_and_cache(
    results: dict[int, dict],
    claim: Claim,
    claim_key: str,
    sources: list[Source],
    search_meta: dict,
    verdict: Verdict,
    search_error: str | None,
) -> None:
    logger.info(
        "VERDICT | claim_id=%d | verdict=%s | confidence=%.2f",
        claim.claim_id, verdict.verdict, verdict.confidence,
    )

    # Component scores behind the deterministic confidence (see scoring.py).
    bd = verdict.confidence_breakdown or {}
    comps = bd.get("components")
    if comps:
        logger.info(
            "CONFIDENCE | claim_id=%d | relevance=%.2f | credibility=%.2f | "
            "strength=%.2f | agreement=%.2f | dominance=%.2f | weighted_sum=%.2f | "
            "conflict_capped=%s | final=%.2f",
            claim.claim_id, comps["relevance"], comps["credibility"],
            comps["strength"], comps["agreement"], bd["dominance"],
            bd["weighted_sum"], bd["conflict_capped"], verdict.confidence,
        )
    else:
        logger.info(
            "CONFIDENCE | claim_id=%d | no component scores | reason=%s | final=%.2f",
            claim.claim_id, bd.get("reason", "n/a"), verdict.confidence,
        )

    results[claim.claim_id] = {
        "sources": sources,
        "search_meta": search_meta,
        "verdict": verdict,
        "from_cache": False,
        "search_error": search_error,
    }

    cache_utils.set(
        "claim_results",
        claim_key,
        {
            "sources": [s.model_dump() for s in sources],
            "search_meta": search_meta,
            "verdict": verdict.model_dump(),
        },
    )