"""
Batched LLM fact-checking.

Claims that DO have sources are sent to the LLM in FIXED BATCHES of BATCH_SIZE
claims per Groq call via fact_check_claims_batch(). Callers (see
factcheck.pipeline) group claims into batches and resolve no-source claims
separately with fact_check_no_sources().
"""

from factcheck import config, llm
from factcheck.logging_utils import logger
from factcheck.models import Source, Verdict
from factcheck.scoring import nli_for_claims, parse_assessments
from factcheck.scoring.nli import counted_indices
from factcheck.verdicts.parsing import (
    fallback_invalid_verdict,
    match_objects_to_items,
    parse_batch_response,
    validate_and_verify,
)
from factcheck.verdicts.prompt import SYSTEM_PROMPT, build_batch_user_prompt

BATCH_SIZE = config.BATCH_SIZE


def _call_llm_batch(user_prompt: str, batch_size: int) -> str:
    return llm.chat(
        SYSTEM_PROMPT,
        user_prompt,
        model=config.VERDICT_MODEL,
        stage="fact_check_batch",
        # Per-claim budget covers the verdict text plus the per-source
        # assessment list (up to 5 small objects per claim).
        max_tokens=min(4000, 1600 * batch_size),
        log_fields=f" | batch_size={batch_size}",
    )


def _request_parsed_list(user_prompt: str, batch_size: int) -> list | None:
    """Calls the model and parses its reply; one stricter retry if the JSON is bad."""
    raw_output = llm.strip_code_fences(_call_llm_batch(user_prompt, batch_size))
    parsed_list = parse_batch_response(raw_output)

    if parsed_list is None:
        logger.warning("BATCH parse failed, retrying | batch_size=%d", batch_size)
        raw_output = llm.strip_code_fences(
            _call_llm_batch(
                user_prompt + "\n\nREMINDER: Return ONLY a valid JSON array, nothing else.",
                batch_size,
            )
        )
        parsed_list = parse_batch_response(raw_output)

    return parsed_list


def fact_check_claims_batch(items: list[tuple[str, list[Source]]]) -> list[Verdict]:
    """
    Sends up to BATCH_SIZE claims (each with its own non-empty source list)
    to the LLM in a SINGLE Groq call, and returns one Verdict per item, in
    the same order as `items`.

    items: list of (claim_text, sources). Every `sources` list must be
    non-empty — callers should resolve no-source claims separately via
    fact_check_no_sources() before calling this.
    """
    if not items:
        return []

    user_prompt = build_batch_user_prompt(items)
    parsed_list = _request_parsed_list(user_prompt, len(items))

    if parsed_list is None:
        logger.error("BATCH failed after retry | batch_size=%d", len(items))
        return [fallback_invalid_verdict() for _ in items]

    matched = match_objects_to_items(parsed_list, len(items))

    # One batched local NLI pass over every counted (claim, snippet) pair.
    requests = []
    for (claim_text, sources), obj in zip(items, matched):
        idxs = []
        if obj is not None and obj.get("verdict") != "UNVERIFIABLE":
            idxs = counted_indices(parse_assessments(obj.get("source_assessments"), len(sources)),
                       obj.get("verdict"), config.NLI_MAX_PAIRS_PER_CLAIM)
        requests.append((claim_text, sources, idxs))
    nli_by_item = nli_for_claims(requests)

    verdicts = []
    for i, (_claim_text, sources) in enumerate(items):
        obj = matched[i]
        if obj is None:
            logger.warning("BATCH missing result | claim_index=%d", i)
            verdicts.append(fallback_invalid_verdict())
        else:
            verdicts.append(validate_and_verify(obj, sources, nli_by_item[i]))

    return verdicts
