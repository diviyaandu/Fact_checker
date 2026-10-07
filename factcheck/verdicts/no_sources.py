"""Verdicts for claims whose evidence search returned zero usable sources
(no LLM call is made)."""

from factcheck.models import Verdict


def fact_check_no_sources(search_meta: dict | None) -> Verdict:
    """
    Resolves a claim's verdict when evidence search returned zero sources.

    - If the search completely failed, return UNVERIFIABLE.
    - If the search successfully ran but found zero raw results, return a
      low-confidence FALSE signal (a fixed heuristic, not from the scoring formula).
    - Otherwise (search ran, found results, but none passed the relevance
      filter in evidence/search.py) return UNVERIFIABLE.
    """
    total_raw = (search_meta or {}).get("total_raw_results", 0)
    queries_succeeded = (search_meta or {}).get("queries_succeeded", 0)

    # Search did not successfully run at all.
    # This is a technical problem, NOT evidence that the claim is false.
    if queries_succeeded == 0:
        return Verdict(
            verdict="UNVERIFIABLE",
            confidence=0.0,
            summary="The evidence search failed, so this claim couldn't be checked.",
            explanation=(
                "A technical error prevented the search from running, so no "
                "evidence was gathered either way. This is a system issue, "
                "not a finding about the claim itself — it should be "
                "re-checked once search is working."
            ),
            cited_sources=[],
        )

    # Search successfully ran but returned absolutely nothing.
    if total_raw == 0:
        return Verdict(
            verdict="FALSE",
            confidence=0.45,
            summary=(
                "This looks made up or misnamed — a real search for the exact "
                "name (and likely misspellings) found nothing."
            ),
            explanation=(
                "A web search — including a check for alternate spellings, "
                "in case the name was mis-transcribed — returned no results "
                "at all. Real named places, people, or studies almost always "
                "show up in at least one search result somewhere, so a total "
                "blank is a meaningful sign the name may be wrong or invented. "
                "This isn't certain: very obscure local names are occasionally "
                "missing from the indexed web, so treat this as a strong hint "
                "rather than a confirmed fact."
            ),
            cited_sources=[],
            # Fixed heuristic (no sources to score) — not from the formula.
            confidence_breakdown={"reason": "heuristic_no_search_results", "confidence": 0.45},
        )

    # Search worked and found results, but none passed the relevance filter.
    return Verdict(
        verdict="UNVERIFIABLE",
        confidence=0.0,
        summary="There isn't enough evidence available to check this claim.",
        explanation=(
            "No sources addressing this specific claim were found, "
            "so it cannot be verified either way."
        ),
        cited_sources=[],
    )
