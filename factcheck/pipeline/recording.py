"""Recording a finished verdict: logging it (with the scoring components) and
writing it to the results dict and the disk cache."""

from factcheck import cache
from factcheck.logging_utils import logger
from factcheck.models import Claim, Source, Verdict


def log_confidence(claim_id: int, verdict: Verdict) -> None:
    """Logs the component scores behind the deterministic confidence (see scoring/)."""
    bd = verdict.confidence_breakdown or {}
    comps = bd.get("components")

    if comps:
        logger.info(
            "CONFIDENCE | claim_id=%d | relevance=%.2f | credibility=%.2f | "
            "strength=%.2f | agreement=%.2f | dominance=%.2f | weighted_sum=%.2f | "
            "conflict_capped=%s | final=%.2f",
            claim_id, comps["relevance"], comps["credibility"],
            comps["strength"], comps["agreement"], bd["dominance"],
            bd["weighted_sum"], bd["conflict_capped"], verdict.confidence,
        )
    else:
        logger.info(
            "CONFIDENCE | claim_id=%d | no component scores | reason=%s | final=%.2f",
            claim_id, bd.get("reason", "n/a"), verdict.confidence,
        )


def record_and_cache(
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
    log_confidence(claim.claim_id, verdict)

    results[claim.claim_id] = {
        "sources": sources,
        "search_meta": search_meta,
        "verdict": verdict,
        "from_cache": False,
        "search_error": search_error,
    }

    cache.set(
        "claim_results",
        claim_key,
        {
            "sources": [s.model_dump() for s in sources],
            "search_meta": search_meta,
            "verdict": verdict.model_dump(),
        },
    )
