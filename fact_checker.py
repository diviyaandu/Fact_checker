"""
fact_checker.py

Takes claims + retrieved evidence sources and produces evidence-grounded
verdicts.

Claims that have zero retrieved sources are resolved locally with
fact_check_no_sources() — no LLM call needed.

Claims that DO have sources are sent to the LLM in FIXED BATCHES of
BATCH_SIZE claims per Groq call via fact_check_claims_batch(). Callers
(see pipeline.get_claim_verdicts) are responsible for grouping claims into
batches and for calling fact_check_no_sources() separately for claims with
no sources — this module just does the batch call and the no-source logic.
"""

import os
import json
from typing import Literal

from pydantic import BaseModel, ValidationError
from groq import Groq
from dotenv import load_dotenv

from search import Source
import time
from logging_utils import logger
from scoring import parse_assessments, compute_confidence

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

BATCH_SIZE = 2

VERDICTS = [
    "TRUE",
    "FALSE",
    "PARTIALLY TRUE",
    "MISLEADING",
    "UNVERIFIABLE"
]


class CitedSource(BaseModel):
    publisher: str
    url: str
    title: str
    relevant_text: str


class Verdict(BaseModel):
    verdict: Literal[
        "TRUE",
        "FALSE",
        "PARTIALLY TRUE",
        "MISLEADING",
        "UNVERIFIABLE"
    ]
    # Computed deterministically in scoring.py from the LLM's structured
    # per-source assessments — the LLM never supplies this number.
    confidence: float
    summary: str
    explanation: str
    cited_sources: list[CitedSource]
    # Component scores behind `confidence` (None for heuristic/fallback
    # verdicts and for results cached before this field existed).
    confidence_breakdown: dict | None = None


SYSTEM_PROMPT = """You are an evidence-based fact-checking assistant writing for a
general audience, not other researchers.

You will be given MULTIPLE CLAIMS in a single request. Each claim starts with
a line "=== CLAIM <index> ===" followed by its claim text and its OWN numbered
list of SOURCES (title, publisher, url, credibility tier, snippet). Evaluate
every claim INDEPENDENTLY: use only that claim's own sources, and never let
evidence from one claim influence the verdict for another claim.

For EACH claim, do the following:

STEP 1 — ASSESS EVERY SOURCE:
For EVERY source listed under a claim (use its [index] number), judge it
against the claim using only that source's own snippet:

- "relevance" (integer 0-3): how directly the source addresses the claim's
  specific subject matter.
    0 = unrelated, or only shares a keyword
    1 = tangential / general background only
    2 = addresses the subject but only part of the specific assertion
    3 = directly addresses the specific assertion
- "stance" (one of "supports", "contradicts", "partial", "neutral"): what the
  snippet says about the claim AS STATED.
    supports    = confirms the claim
    contradicts = disputes the claim
    partial     = confirms some elements but not others (overstated, missing
                  context, only true under conditions)
    neutral     = says nothing that bears on whether the claim is correct
- "directness" (integer 0-3): how explicitly the snippet states the fact at
  issue.
    0 = nothing bearing on the claim
    1 = only implied, or general background
    2 = stated clearly, but not with the claim's exact specifics
    3 = explicitly states the specific fact, figure, date or scope in the claim
        (or its direct negation)

Do NOT weigh a source up or down for its publisher or credibility tier — that
is handled separately. Judge only what the snippet says.

You MAY use general background/context explained within a relevant source
(e.g. a source explaining plate tectonics) to reason about whether the claim
is consistent with that established science — as long as the source is
cited. This is synthesis of what the source actually says, not invention of
new facts.

STEP 2 — VERDICT:
Using ONLY the sources you judged relevant for THAT claim, determine the
verdict:

- TRUE: relevant evidence clearly confirms the claim as stated
- FALSE: relevant evidence clearly contradicts the claim
- PARTIALLY TRUE: some elements are accurate but the claim overstates, omits
  key context, or is only true under specific conditions
- MISLEADING: technically has some basis but creates a false impression
- UNVERIFIABLE: no relevant sources were found, or they don't provide enough
  information to judge

CRITICAL RULES:

1. Only use information present in that claim's own provided sources. Never
   use outside knowledge you may have about the topic, and never borrow a
   source from a different claim.

2. Only cite sources you judged relevant for that specific claim. Never cite
   an irrelevant source.

3. Never invent a URL, title, or publisher — only use exactly what was
   provided for that claim.

4. If zero sources are relevant for a claim, that claim's verdict MUST be
   UNVERIFIABLE.

5. Do NOT output any confidence or probability number. Confidence is
   computed separately from your per-source assessments, so make those
   assessments honest and precise. When sources disagree, give the verdict
   the best-supported reading, with Tier 1-2 sources (government/scientific)
   outweighing Tier 4-5 sources.

6. WRITING STYLE:
   - "summary": ONE short, plain sentence a general reader gets in 3 seconds.
     Start with the bottom line.
   - "explanation": 2-4 short sentences MAX, plain conversational language,
     no jargon.

OUTPUT FORMAT:
Return ONLY a valid JSON array, no markdown, no code fences. The array must
contain exactly one object per claim you were given (in any order), and each
object MUST include "claim_index" set to the number from that claim's
"=== CLAIM <index> ===" header, in this exact shape:

[
  {
    "claim_index": 0,
    "verdict": "...",
    "source_assessments": [
      {"source_index": 0, "relevance": 0, "stance": "neutral", "directness": 0}
    ],
    "summary": "...",
    "explanation": "...",
    "cited_sources": [
      {
        "publisher": "...",
        "url": "...",
        "title": "...",
        "relevant_text": "..."
      }
    ]
  }
]

"source_assessments" must contain one entry for EVERY source of that claim,
including irrelevant ones (relevance 0, stance "neutral", directness 0).
"""


def _build_batch_user_prompt(items: list[tuple[str, list[Source]]]) -> str:
    blocks = []

    for idx, (claim_text, sources) in enumerate(items):
        lines = [f"=== CLAIM {idx} ===", f"CLAIM TEXT:\n{claim_text}\n", "SOURCES:"]

        for i, src in enumerate(sources):
            lines.append(
                f"\n[{i}] Publisher: {src.publisher} "
                f"(Credibility Tier: {src.credibility_tier})\n"
                f"Title: {src.title}\n"
                f"URL: {src.url}\n"
                f"Snippet: {src.relevant_text}"
            )

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks)


def _strip_code_fences(text: str) -> str:
    text = text.strip()

    if text.startswith("```"):
        lines = [
            line
            for line in text.split("\n")
            if not line.strip().startswith("```")
        ]
        text = "\n".join(lines)

    return text.strip()


def _call_llm_batch(user_prompt: str, batch_size: int) -> str:
    logger.info(
        "GROQ call | stage=fact_check_batch | model=%s | batch_size=%d | input_chars=%d",
        MODEL,
        batch_size,
        len(user_prompt),
    )

    start = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            # Slightly larger per-claim budget: output now also carries a
            # per-source assessment list (up to 5 small objects per claim).
            max_tokens=min(4000, 1600 * batch_size),
        )

        elapsed = time.time() - start

        logger.info(
            "GROQ success | stage=fact_check_batch | batch_size=%d | duration=%.2fs",
            batch_size,
            elapsed,
        )

        return response.choices[0].message.content.strip()

    except Exception:
        logger.exception("GROQ failure | stage=fact_check_batch | batch_size=%d", batch_size)
        raise


def _fallback_invalid_verdict() -> Verdict:
    return Verdict(
        verdict="UNVERIFIABLE",
        confidence=0.0,
        summary="The fact-checking step failed to produce a valid result.",
        explanation="The fact-checking model returned an invalid response format.",
        cited_sources=[],
    )


def _validate_and_verify(parsed_obj: dict, sources: list[Source]) -> Verdict:
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
        return _fallback_invalid_verdict()

    if "confidence" in parsed_obj:
        logger.debug("SCORE | ignoring LLM-supplied confidence=%r", parsed_obj["confidence"])

    # If the model gave a definite verdict but none of its citations matched
    # the URLs we actually provided for THIS claim, don't trust the verdict.
    if verdict.verdict != "UNVERIFIABLE" and not verdict.cited_sources:
        return _unverifiable_no_evidence(
            verdict, "No verifiably relevant evidence was retrieved to support "
            "a verdict on this claim.", "no_valid_citations",
        )

    # Deterministic confidence from the LLM's structured per-source factors.
    if verdict.verdict != "UNVERIFIABLE":
        assessments = parse_assessments(parsed_obj.get("source_assessments"), len(sources))
        confidence, breakdown = compute_confidence(verdict.verdict, assessments, sources)

        if breakdown.get("no_aligned_evidence"):
            # Verdict isn't backed by any usable assessed evidence (e.g. TRUE
            # but no source actually "supports", or assessments missing).
            return _unverifiable_no_evidence(
                verdict, "The retrieved sources didn't clearly back up any verdict "
                "on this claim.", "no_aligned_evidence", breakdown,
            )

        verdict.confidence = confidence
        verdict.confidence_breakdown = breakdown

    # UNVERIFIABLE should always have zero confidence.
    if verdict.verdict == "UNVERIFIABLE":
        verdict.confidence = 0.0

    return verdict


def _unverifiable_no_evidence(
    verdict: Verdict, explanation: str, reason: str, breakdown: dict | None = None
) -> Verdict:
    logger.warning("SCORE | verdict downgraded to UNVERIFIABLE | was=%s | reason=%s", verdict.verdict, reason)
    verdict.verdict = "UNVERIFIABLE"
    verdict.confidence = 0.0
    verdict.summary = "There isn't enough verifiable evidence to support a verdict."
    verdict.explanation = explanation
    verdict.confidence_breakdown = breakdown or {"reason": reason}
    return verdict


def _parse_batch_response(raw_output: str) -> list | None:
    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, list):
        return None

    return parsed


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

    user_prompt = _build_batch_user_prompt(items)
    raw_output = _call_llm_batch(user_prompt, len(items))
    raw_output = _strip_code_fences(raw_output)

    parsed_list = _parse_batch_response(raw_output)

    if parsed_list is None:
        logger.warning("BATCH parse failed, retrying | batch_size=%d", len(items))
        raw_output = _call_llm_batch(
            user_prompt + "\n\nREMINDER: Return ONLY a valid JSON array, nothing else.",
            len(items),
        )
        raw_output = _strip_code_fences(raw_output)
        parsed_list = _parse_batch_response(raw_output)

    if parsed_list is None:
        logger.error("BATCH failed after retry | batch_size=%d", len(items))
        return [_fallback_invalid_verdict() for _ in items]

    # Map parsed objects back to items by claim_index; fall back to
    # positional order for anything malformed or missing an index.
    matched: list[dict | None] = [None] * len(items)
    leftovers = []

    for obj in parsed_list:
        if not isinstance(obj, dict):
            continue

        idx = obj.get("claim_index")
        if isinstance(idx, int) and 0 <= idx < len(items) and matched[idx] is None:
            matched[idx] = obj
        else:
            leftovers.append(obj)

    leftover_i = 0
    for i in range(len(items)):
        if matched[i] is None and leftover_i < len(leftovers):
            matched[i] = leftovers[leftover_i]
            leftover_i += 1

    verdicts = []
    for i, (_claim_text, sources) in enumerate(items):
        obj = matched[i]
        if obj is None:
            logger.warning("BATCH missing result | claim_index=%d", i)
            verdicts.append(_fallback_invalid_verdict())
        else:
            verdicts.append(_validate_and_verify(obj, sources))

    return verdicts


def fact_check_no_sources(search_meta: dict | None) -> Verdict:
    """
    Resolves a claim's verdict when evidence search returned zero sources.
    No LLM call is made.

    - If the search completely failed, return UNVERIFIABLE.
    - If the search successfully ran but found zero raw results, return a
      low-confidence FALSE signal.
    - Otherwise (search ran, found results, but none passed the relevance
      filter in search.py) return UNVERIFIABLE.
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