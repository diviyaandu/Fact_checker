"""
Evidence + verdict stage.

Verdict generation batches the claims that need an LLM call (see
verdicts.fact_check_claims_batch) BATCH_SIZE at a time. Evidence search for
every uncached claim happens first, so the batches only ever contain claims
that already have their sources ready.
"""

import math

from factcheck import cache
from factcheck.evidence import search_evidence_for_claim
from factcheck.logging_utils import logger
from factcheck.models import Claim, Source, Verdict
from factcheck.pipeline.cache_check import claim_cache_key
from factcheck.pipeline.recording import record_and_cache
from factcheck.verdicts import BATCH_SIZE, fact_check_claims_batch, fact_check_no_sources

# (claim, cache key, sources, search_meta) — a claim whose sources are ready
_Ready = tuple[Claim, str, list[Source], dict]


def get_claim_verdicts(claims: list[Claim], use_cache: bool) -> dict[int, dict]:
    """
    Resolves sources + verdict for every claim.

    Returns: {claim_id: {"sources": [Source], "search_meta": dict,
                          "verdict": Verdict, "from_cache": bool,
                          "search_error": str | None}}

    Flow:
      1. Check the per-claim cache first; cached claims need no further work.
      2. For every uncached claim, run evidence search (Tavily) up front, for
         ALL of them, before any LLM call is made.
      3. Claims that came back with zero sources are resolved locally
         (no LLM call needed) via verdicts.fact_check_no_sources.
      4. Remaining claims (those with sources) are sent to the LLM in fixed
         batches of BATCH_SIZE via verdicts.fact_check_claims_batch. The final
         batch may contain fewer than BATCH_SIZE claims (as few as 1).
    """
    results: dict[int, dict] = {}

    pending = _load_cached(claims, use_cache, results)
    if not pending:
        return results

    evidence = _search_all(pending)
    needs_llm = _resolve_without_llm(pending, evidence, results)
    _resolve_with_llm(needs_llm, results)

    return results


# ---- step 1 ---------------------------------------------------------------

def _load_cached(claims: list[Claim], use_cache: bool, results: dict) -> list[tuple[Claim, str]]:
    """Fills `results` from the cache; returns the (claim, key) pairs still to do."""
    pending: list[tuple[Claim, str]] = []

    for claim in claims:
        claim_key = claim_cache_key(claim.claim_text)
        cached_result = cache.get("claim_results", claim_key) if use_cache else None

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

    return pending


# ---- step 2 ---------------------------------------------------------------

def _search_all(pending: list[tuple[Claim, str]]) -> dict[int, tuple[list[Source], dict, str | None]]:
    """Evidence search for every pending claim -> {claim_id: (sources, meta, error)}."""
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

    return evidence


# ---- step 3 ---------------------------------------------------------------

def _resolve_without_llm(pending, evidence, results: dict) -> list[_Ready]:
    """Records verdicts for claims with no sources; returns the rest (need the LLM)."""
    needs_llm: list[_Ready] = []

    for claim, claim_key in pending:
        found_sources, search_meta, search_error = evidence[claim.claim_id]

        if found_sources:
            needs_llm.append((claim, claim_key, found_sources, search_meta))
        else:
            verdict = fact_check_no_sources(search_meta)
            record_and_cache(results, claim, claim_key, found_sources, search_meta, verdict, search_error)

    return needs_llm


# ---- step 4 ---------------------------------------------------------------

def _resolve_with_llm(needs_llm: list[_Ready], results: dict) -> None:
    """Batches the LLM calls, BATCH_SIZE claims per Groq call."""
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
            record_and_cache(results, claim, claim_key, sources, search_meta, verdict, None)
