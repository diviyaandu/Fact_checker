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
    confidence: float
    summary: str
    explanation: str
    cited_sources: list[CitedSource]


SYSTEM_PROMPT = """You are an evidence-based fact-checking assistant writing for a
general audience, not other researchers.

You will be given MULTIPLE CLAIMS in a single request. Each claim starts with
a line "=== CLAIM <index> ===" followed by its claim text and its OWN numbered
list of SOURCES (title, publisher, url, credibility tier, snippet). Evaluate
every claim INDEPENDENTLY: use only that claim's own sources, and never let
evidence from one claim influence the verdict for another claim.

For EACH claim, do the following:

STEP 1 — RELEVANCE CHECK:
For each of that claim's sources, judge whether it actually addresses the
claim's specific subject matter. Ignore sources that only share a keyword
but don't substantively address the claim's actual content.

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

5. confidence (0.0-1.0) reflects how strongly the RELEVANT evidence supports
   your verdict — not how "true" the claim is. Tier 1-2 sources
   (government/scientific) should carry more weight than Tier 4-5 sources
   when they disagree.

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
    "confidence": 0.0,
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
            max_tokens=min(4000, 1300 * batch_size),
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
        verdict = Verdict(
            verdict=parsed_obj.get("verdict", "UNVERIFIABLE"),
            confidence=float(parsed_obj.get("confidence", 0.0)),
            summary=parsed_obj.get("summary", ""),
            explanation=parsed_obj.get("explanation", ""),
            cited_sources=verified_citations,
        )
    except (ValidationError, ValueError, TypeError):
        return _fallback_invalid_verdict()

    # If the model gave a definite verdict but none of its citations matched
    # the URLs we actually provided for THIS claim, don't trust the verdict.
    if verdict.verdict != "UNVERIFIABLE" and not verdict.cited_sources:
        verdict.verdict = "UNVERIFIABLE"
        verdict.confidence = 0.0
        verdict.summary = "There isn't enough verifiable evidence to support a verdict."
        verdict.explanation = (
            "No verifiably relevant evidence was retrieved to support "
            "a verdict on this claim."
        )

    # UNVERIFIABLE should always have zero confidence.
    if verdict.verdict == "UNVERIFIABLE":
        verdict.confidence = 0.0

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