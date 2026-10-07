"""
Extracts up to MAX_CLAIMS checkable factual claims from a transcript using an
LLM (Groq).

Design principle: the LLM never invents timestamps. It must reference numbered
transcript segments, and we look up the real start/end times ourselves. This
prevents timestamp hallucination.
"""

import json

from pydantic import ValidationError

from factcheck import config, llm
from factcheck.claims.prompt import SYSTEM_PROMPT
from factcheck.models import ALLOWED_CATEGORIES, Claim


def _build_numbered_transcript(segments: list[dict]) -> str:
    return "\n".join(f"[{i}] {seg['text']}" for i, seg in enumerate(segments))


def _call_llm(numbered_transcript: str) -> str:
    return llm.chat(
        SYSTEM_PROMPT,
        numbered_transcript,
        model=config.CLAIMS_MODEL,
        stage="claim_extraction",
        max_tokens=1500,
    )


def _load_candidates(numbered_transcript: str):
    """Asks the model for claims and parses its JSON; one stricter retry, then
    gives up cleanly (returns None) rather than crash the app."""
    raw_output = llm.strip_code_fences(_call_llm(numbered_transcript))

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    raw_output = _call_llm(
        numbered_transcript + "\n\nREMINDER: Return ONLY valid JSON array, nothing else."
    )
    try:
        return json.loads(llm.strip_code_fences(raw_output))
    except json.JSONDecodeError:
        return None


def _default_queries(claim_text: str) -> list[str]:
    # Fallback so evidence search always has something usable even if the model
    # skipped this field on a given claim.
    return [
        claim_text,
        f"{claim_text} fact check",
        f"is it true that {claim_text.lower()}",
    ]


def _build_claim(candidate, segments: list[dict], claim_id: int) -> Claim | None:
    """Validates one model-proposed claim; None means 'reject it'."""
    if not isinstance(candidate, dict):
        return None

    segment_indices = candidate.get("segment_indices", [])
    if not segment_indices or not all(isinstance(i, int) for i in segment_indices):
        return None

    # Validate every referenced index actually exists in our transcript;
    # reject the claim rather than guess timestamps.
    if any(i < 0 or i >= len(segments) for i in segment_indices):
        return None

    if candidate.get("category") not in ALLOWED_CATEGORIES:
        return None

    referenced = [segments[i] for i in segment_indices]
    real_start = min(seg["start_str"] for seg in referenced)
    real_end = max(seg["end_str"] for seg in referenced)

    claim_text = candidate.get("claim_text", "")
    queries = candidate.get("search_queries", [])
    if not (isinstance(queries, list) and all(isinstance(q, str) and q.strip() for q in queries)):
        queries = []
    if not queries:
        queries = _default_queries(claim_text)

    try:
        return Claim(
            claim_id=claim_id,
            claim_text=claim_text,
            start_time=real_start,
            end_time=real_end,
            category=candidate.get("category"),
            importance=candidate.get("importance", "medium"),
            search_queries=queries,
        )
    except ValidationError:
        return None  # skip malformed claim rather than crash


def extract_claims(segments: list[dict]) -> list[Claim]:
    """
    segments: list of {start, end, start_str, end_str, text} from factcheck.transcript
    Returns: list of validated Claim objects (max MAX_CLAIMS) with real timestamps.
    """
    if not segments:
        return []

    candidates = _load_candidates(_build_numbered_transcript(segments))

    if not isinstance(candidates, list):
        return []

    valid_claims: list[Claim] = []

    for candidate in candidates[: config.MAX_CLAIMS]:  # hard cap, even if the model returns more
        claim = _build_claim(candidate, segments, claim_id=len(valid_claims) + 1)
        if claim is not None:
            valid_claims.append(claim)

    return valid_claims
