"""
fact_checker.py

Takes a claim + retrieved evidence sources, and produces an evidence-grounded verdict.
"""

import os
import json
from typing import Literal

from pydantic import BaseModel, ValidationError
from groq import Groq
from dotenv import load_dotenv

from search import Source

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

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

You will be given a CLAIM and a numbered list of SOURCES (title, publisher, url,
credibility tier, snippet).

STEP 1 — RELEVANCE CHECK:
For each source, judge whether it actually addresses the claim's specific subject
matter. Ignore sources that only share a keyword but don't substantively address
the claim's actual content.

You MAY use general background/context explained within a relevant source
(e.g. a source explaining plate tectonics) to reason about whether the claim is
consistent with that established science — as long as the source is cited. This
is synthesis of what the source actually says, not invention of new facts.

STEP 2 — VERDICT:
Using ONLY the sources you judged relevant, determine the verdict:

- TRUE: relevant evidence clearly confirms the claim as stated
- FALSE: relevant evidence clearly contradicts the claim
- PARTIALLY TRUE: some elements are accurate but the claim overstates, omits
  key context, or is only true under specific conditions
- MISLEADING: technically has some basis but creates a false impression
- UNVERIFIABLE: no relevant sources were found, or they don't provide enough
  information to judge

CRITICAL RULES:

1. Only use information present in the provided sources. Never use outside
   knowledge you may have about the topic.

2. Only cite sources you judged relevant. Never cite an irrelevant source.

3. Never invent a URL, title, or publisher — only use exactly what was provided.

4. If zero sources are relevant, verdict MUST be UNVERIFIABLE (unless the SOURCES
   section explicitly tells you this was a confirmed zero-result search for a
   named entity — follow those specific instructions if given).

5. confidence (0.0-1.0) reflects how strongly the RELEVANT evidence supports your
   verdict — not how "true" the claim is. Tier 1-2 sources (government/scientific)
   should carry more weight than Tier 4-5 sources when they disagree.

6. WRITING STYLE:
   - "summary": ONE short, plain sentence a general reader gets in 3 seconds.
     Start with the bottom line.
   - "explanation": 2-4 short sentences MAX, plain conversational language,
     no jargon.

Return ONLY valid JSON, no markdown, no code fences, in this exact shape:

{
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
"""


def _build_user_prompt(claim_text: str, sources: list[Source]) -> str:
    lines = [f"CLAIM:\n{claim_text}\n\nSOURCES:"]

    for i, src in enumerate(sources):
        lines.append(
            f"\n[{i}] Publisher: {src.publisher} "
            f"(Credibility Tier: {src.credibility_tier})\n"
            f"Title: {src.title}\n"
            f"URL: {src.url}\n"
            f"Snippet: {src.relevant_text}"
        )

    return "\n".join(lines)


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


def _call_llm(user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.1,
        max_tokens=1200,
    )

    return response.choices[0].message.content.strip()


def fact_check_claim(
    claim_text: str,
    sources: list[Source],
    search_meta: dict | None = None
) -> Verdict:

    """
    Returns a Verdict grounded ONLY in the provided sources.

    If no sources were found:
    - If the search completely failed, return UNVERIFIABLE.
    - If the search successfully ran but found zero results, return
      a low-confidence FALSE signal.
    - Otherwise return UNVERIFIABLE.
    """

    # ============================================================
    # NO SOURCES CASE
    # ============================================================
    if not sources:

        total_raw = (search_meta or {}).get(
            "total_raw_results",
            0
        )

        queries_succeeded = (search_meta or {}).get(
            "queries_succeeded",
            0
        )

        # Search did not successfully run at all.
        # This is a technical problem, NOT evidence that the claim is false.
        if queries_succeeded == 0:

            return Verdict(
                verdict="UNVERIFIABLE",
                confidence=0.0,
                summary=(
                    "The evidence search failed, so this claim couldn't be checked."
                ),
                explanation=(
                    "A technical error prevented the search from running, so no "
                    "evidence was gathered either way. This is a system issue, "
                    "not a finding about the claim itself — it should be "
                    "re-checked once search is working."
                ),
                cited_sources=[]
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
                cited_sources=[]
            )

        # Search worked and found results, but none passed the relevance filter.
        return Verdict(
            verdict="UNVERIFIABLE",
            confidence=0.0,
            summary=(
                "There isn't enough evidence available to check this claim."
            ),
            explanation=(
                "No sources addressing this specific claim were found, "
                "so it cannot be verified either way."
            ),
            cited_sources=[]
        )

    # ============================================================
    # SOURCES EXIST — SEND THEM TO THE LLM
    # ============================================================

    user_prompt = _build_user_prompt(
        claim_text,
        sources
    )

    raw_output = _call_llm(user_prompt)

    raw_output = _strip_code_fences(raw_output)

    # ============================================================
    # PARSE JSON
    # ============================================================

    try:
        parsed = json.loads(raw_output)

    except json.JSONDecodeError:

        # Retry once if the model didn't return valid JSON.
        raw_output = _call_llm(
            user_prompt
            + "\n\nREMINDER: Return ONLY valid JSON, nothing else."
        )

        raw_output = _strip_code_fences(raw_output)

        try:
            parsed = json.loads(raw_output)

        except json.JSONDecodeError:

            return Verdict(
                verdict="UNVERIFIABLE",
                confidence=0.0,
                summary=(
                    "The fact-checking step failed to produce a valid result."
                ),
                explanation=(
                    "The fact-checking model returned an invalid response format."
                ),
                cited_sources=[]
            )

    # ============================================================
    # VERIFY CITATIONS
    # ============================================================

    real_urls = {
        src.url
        for src in sources
    }

    raw_citations = parsed.get(
        "cited_sources",
        []
    )

    verified_citations = []

    for citation in raw_citations:

        if (
            isinstance(citation, dict)
            and citation.get("url") in real_urls
        ):
            verified_citations.append(citation)

    # ============================================================
    # VALIDATE VERDICT
    # ============================================================

    try:

        verdict = Verdict(
            verdict=parsed.get(
                "verdict",
                "UNVERIFIABLE"
            ),

            confidence=float(
                parsed.get(
                    "confidence",
                    0.0
                )
            ),

            summary=parsed.get(
                "summary",
                ""
            ),

            explanation=parsed.get(
                "explanation",
                ""
            ),

            cited_sources=verified_citations
        )

    except (
        ValidationError,
        ValueError,
        TypeError
    ):

        return Verdict(
            verdict="UNVERIFIABLE",
            confidence=0.0,
            summary=(
                "The fact-checking step failed to produce a valid result."
            ),
            explanation=(
                "Could not validate the fact-checking model's output."
            ),
            cited_sources=[]
        )

    # ============================================================
    # SAFETY CHECK
    # ============================================================

    # If the model gave a definite verdict but none of its citations
    # matched the URLs we actually provided, don't trust the verdict.
    if (
        verdict.verdict != "UNVERIFIABLE"
        and not verdict.cited_sources
    ):

        verdict.verdict = "UNVERIFIABLE"

        verdict.confidence = 0.0

        verdict.summary = (
            "There isn't enough verifiable evidence to support a verdict."
        )

        verdict.explanation = (
            "No verifiably relevant evidence was retrieved to support "
            "a verdict on this claim."
        )

    # UNVERIFIABLE should always have zero confidence.
    if verdict.verdict == "UNVERIFIABLE":
        verdict.confidence = 0.0

    return verdict