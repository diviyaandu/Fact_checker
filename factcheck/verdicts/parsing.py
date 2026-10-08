"""
Turns the LLM's JSON into trustworthy Verdict objects.

  * parse_batch_response()      JSON text -> list (or None if unusable)
  * match_objects_to_items()    lines each parsed object up with the claim it answers
  * validate_and_verify()       one parsed object -> Verdict: only real citations,
                                deterministic confidence, safe downgrades
"""

import json

from pydantic import ValidationError

from factcheck.logging_utils import logger
from factcheck.models import Source, Verdict
from factcheck.scoring import compute_confidence, parse_assessments


def fallback_invalid_verdict() -> Verdict:
    return Verdict(
        verdict="UNVERIFIABLE",
        confidence=0.0,
        summary="The fact-checking step failed to produce a valid result.",
        explanation="The fact-checking model returned an invalid response format.",
        cited_sources=[],
    )


def unverifiable_no_evidence(
    verdict: Verdict, explanation: str, reason: str, breakdown: dict | None = None
) -> Verdict:
    """Downgrades a verdict the evidence doesn't back up to UNVERIFIABLE (confidence 0)."""
    logger.warning(
        "SCORE | verdict downgraded to UNVERIFIABLE | was=%s | reason=%s", verdict.verdict, reason
    )
    verdict.verdict = "UNVERIFIABLE"
    verdict.confidence = 0.0
    verdict.summary = "There isn't enough verifiable evidence to support a verdict."
    verdict.explanation = explanation
    verdict.confidence_breakdown = breakdown or {"reason": reason}
    return verdict


def validate_and_verify(parsed_obj: dict, sources: list[Source], nli_results: dict | None = None) -> Verdict:
    """Builds a Verdict from one parsed LLM object, restricted to citations
    that actually match this claim's own retrieved sources."""

    real_urls = {src.url for src in sources}
    raw_citations = parsed_obj.get("cited_sources", [])

    verified_citations = [
        citation
        for citation in raw_citations
        if isinstance(citation, dict) and citation.get("url") in real_urls
    ]

    try:
        # Any "confidence" the model may still emit is deliberately ignored;
        # the real value is computed below by scoring.compute_confidence.
        verdict = Verdict(
            verdict=parsed_obj.get("verdict", "UNVERIFIABLE"),
            confidence=0.0,
            summary=parsed_obj.get("summary", ""),
            explanation=parsed_obj.get("explanation", ""),
            cited_sources=verified_citations,
        )
    except (ValidationError, ValueError, TypeError):
        return fallback_invalid_verdict()

    if "confidence" in parsed_obj:
        logger.debug("SCORE | ignoring LLM-supplied confidence=%r", parsed_obj["confidence"])

    # If the model gave a definite verdict but none of its citations matched
    # the URLs we actually provided for THIS claim, don't trust the verdict.
    if verdict.verdict != "UNVERIFIABLE" and not verdict.cited_sources:
        return unverifiable_no_evidence(
            verdict,
            "No verifiably relevant evidence was retrieved to support a verdict on this claim.",
            "no_valid_citations",
        )

    # Deterministic confidence from the LLM's structured per-source factors.
    if verdict.verdict != "UNVERIFIABLE":
        assessments = parse_assessments(parsed_obj.get("source_assessments"), len(sources))
        confidence, breakdown = compute_confidence(verdict.verdict, assessments, sources, nli_results)

        if breakdown.get("no_aligned_evidence"):
            # Verdict isn't backed by any usable assessed evidence (e.g. TRUE
            # but no source actually "supports", or assessments missing).
            return unverifiable_no_evidence(
                verdict,
                "The retrieved sources didn't clearly back up any verdict on this claim.",
                "no_aligned_evidence",
                breakdown,
            )

        verdict.confidence = confidence
        verdict.confidence_breakdown = breakdown

    # UNVERIFIABLE should always have zero confidence.
    if verdict.verdict == "UNVERIFIABLE":
        verdict.confidence = 0.0

    return verdict


def parse_batch_response(raw_output: str) -> list | None:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    return parsed


def match_objects_to_items(parsed_list: list, n_items: int) -> list[dict | None]:
    """Maps parsed objects back to the claims they answer by `claim_index`;
    anything malformed or missing an index fills the remaining slots in order."""
    matched: list[dict | None] = [None] * n_items
    leftovers = []

    for obj in parsed_list:
        if not isinstance(obj, dict):
            continue

        idx = obj.get("claim_index")
        if isinstance(idx, int) and 0 <= idx < n_items and matched[idx] is None:
            matched[idx] = obj
        else:
            leftovers.append(obj)

    leftover_i = 0
    for i in range(n_items):
        if matched[i] is None and leftover_i < len(leftovers):
            matched[i] = leftovers[leftover_i]
            leftover_i += 1

    return matched
