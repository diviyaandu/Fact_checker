"""
search.py

Given a claim, generates targeted search queries and retrieves candidate
evidence from the open web via Tavily, scoring each source's credibility tier.

Also tracks whether the search came back with genuinely zero raw results
(as opposed to results that existed but were irrelevant) — this distinction
matters for claims that assert a specific named entity exists.
"""

import os
import json
from pydantic import BaseModel
from tavily import TavilyClient
from groq import Groq
from dotenv import load_dotenv

from trusted_sources import score_credibility

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

MAX_RESULTS_PER_QUERY = 3
MAX_SOURCES_PER_CLAIM = 5


class Source(BaseModel):
    title: str
    url: str
    publisher: str
    published_date: str | None = None
    relevant_text: str
    credibility_tier: int = 5


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text.strip()


def _generate_queries(claim_text: str, category: str) -> list[str]:
    """
    Ask the LLM for 3 targeted, verification-oriented search queries, focused
    on the SPECIFIC named entities/facts in the claim. If the claim names an
    unusual/uncommon proper noun (a place, person, study), include at least
    one query that searches for that exact name plus one plausible alternate
    spelling — since transcription/subtitle errors are common in short-form
    video (e.g. "Naita-Cainan" is very likely a mis-transcribed real name).
    """
    prompt = f"""Generate exactly 3 short web search queries to verify or refute this claim.
Target the SPECIFIC named entities, places, or measurements in the claim.
If the claim names something unusual (an obscure place/lake/person/statistic):
- one query should search for that exact name alone
- one query should try a plausible alternate spelling or transliteration of
  that name, in case it was mis-transcribed from audio (e.g. "Naita-Cainan"
  could be a garbled version of a real Finnish or Scandinavian place name)

Claim: "{claim_text}"
Category: {category}

Return ONLY a JSON array of 3 short strings, nothing else.
Example: ["query one", "query two", "query three"]
"""
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        raw = _strip_code_fences(response.choices[0].message.content.strip())
        queries = json.loads(raw)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries) and queries:
            return queries[:3]
    except Exception as e:
        print(f"[search] query generation failed, using fallback: {e}")

    return [claim_text, f"{claim_text} fact check", f"is it true that {claim_text.lower()}"]


def _extract_publisher(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).netloc.replace("www.", "")


def search_evidence_for_claim(claim_text: str, category: str) -> tuple[list[Source], dict]:
    """
    Returns (sources, search_meta).

    search_meta tracks:
    - total_raw_results: raw result count across all queries that actually ran
    - queries_succeeded: how many queries executed without throwing an error
    - queries_failed: how many queries errored out (API/network problems)

    This distinction matters: a claim should only be treated as "searched and
    found genuinely nothing" if queries actually ran successfully. If every
    query errored out, that's a search failure, not evidence about the claim.
    """
    queries = _generate_queries(claim_text, category)

    seen_domains = set()
    collected: list[Source] = []
    total_raw_results = 0
    queries_succeeded = 0
    queries_failed = 0

    for query in queries:
        try:
            response = tavily_client.search(
                query=query,
                max_results=MAX_RESULTS_PER_QUERY,
            )
        except Exception as e:
            print(f"[search] Tavily error for query '{query}': {e}")
            queries_failed += 1
            continue

        queries_succeeded += 1
        results = response.get("results", [])
        total_raw_results += len(results)

        for result in results:
            if len(collected) >= MAX_SOURCES_PER_CLAIM:
                continue
            url = result.get("url", "")
            publisher = _extract_publisher(url)
            if publisher in seen_domains:
                continue
            content = result.get("content", "").strip()
            if not content:
                continue
            collected.append(
                Source(
                    title=result.get("title", "Untitled"),
                    url=url,
                    publisher=publisher,
                    published_date=result.get("published_date"),
                    relevant_text=content[:800],
                    credibility_tier=score_credibility(publisher),
                )
            )
            seen_domains.add(publisher)

    collected.sort(key=lambda s: s.credibility_tier)

    search_meta = {
        "queries_used": queries,
        "total_raw_results": total_raw_results,
        "queries_succeeded": queries_succeeded,
        "queries_failed": queries_failed,
        "sources_found": len(collected),
    }
    return collected, search_meta