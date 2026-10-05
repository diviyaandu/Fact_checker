"""
search.py

Given a claim, generates targeted search queries and retrieves candidate
evidence from the open web via Tavily, scoring each source's credibility tier.

Also tracks whether the search came back with genuinely zero raw results
(as opposed to results that existed but were irrelevant) — this distinction
matters for claims that assert a specific named entity exists.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel
from tavily import TavilyClient
from dotenv import load_dotenv

from trusted_sources import score_credibility
from logging_utils import logger

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

MAX_RESULTS_PER_QUERY = 3
MAX_SOURCES_PER_CLAIM = 5
MAX_CONCURRENT_SEARCHES = 2


class Source(BaseModel):
    title: str
    url: str
    publisher: str
    published_date: str | None = None
    relevant_text: str
    credibility_tier: int = 5


def _fallback_queries(claim_text: str) -> list[str]:
    """
    Used only if a claim reaches this function with no queries attached.
    Normal path: queries are generated during claim extraction, so this
    function no longer makes its own LLM call.
    """
    return [
        claim_text,
        f"{claim_text} fact check",
        f"is it true that {claim_text.lower()}",
    ]


def _extract_publisher(url: str) -> str:
    from urllib.parse import urlparse
    return urlparse(url).netloc.replace("www.", "")


def _search_tavily(query: str) -> tuple[str, list[dict]]:
    """Run one Tavily search and return the query plus raw results."""
    logger.info(
        "TAVILY call | query=%s | max_results=%d",
        query,
        MAX_RESULTS_PER_QUERY,
    )

    start = time.time()

    try:
        response = tavily_client.search(
            query=query,
            max_results=MAX_RESULTS_PER_QUERY,
        )

        elapsed = time.time() - start
        results = response.get("results", [])

        logger.info(
            "TAVILY success | query=%s | results=%d | duration=%.2fs",
            query,
            len(results),
            elapsed,
        )

        return query, results

    except Exception:
        logger.exception("TAVILY failure | query=%s", query)
        return query, []


def search_evidence_for_claim(
    claim_text: str,
    category: str,
    queries: list[str] | None = None,
) -> tuple[list[Source], dict]:
    """
    Returns (sources, search_meta).

    queries: the 3 search queries generated during claim extraction.
    If omitted, cheap template queries are used instead.

    Tavily searches run concurrently with a small worker pool to reduce
    latency while remaining conservative with free-tier rate limits.
    """
    queries = queries or _fallback_queries(claim_text)

    seen_domains = set()
    collected: list[Source] = []
    total_raw_results = 0
    queries_succeeded = 0
    queries_failed = 0

    results_by_query = {}

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_SEARCHES) as executor:
        futures = {
            executor.submit(_search_tavily, query): query
            for query in queries
        }

        for future in as_completed(futures):
            query = futures[future]

            try:
                returned_query, results = future.result()
                results_by_query[returned_query] = results

                if results:
                    queries_succeeded += 1
                else:
                    queries_succeeded += 1

                total_raw_results += len(results)

            except Exception:
                logger.exception("TAVILY worker failure | query=%s", query)
                queries_failed += 1

    # Process results in original query order for deterministic output.
    for query in queries:
        results = results_by_query.get(query, [])

        for result in results:
            if len(collected) >= MAX_SOURCES_PER_CLAIM:
                break

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

    logger.info(
        "SEARCH complete | queries=%d | succeeded=%d | failed=%d | raw_results=%d | sources=%d",
        len(queries),
        queries_succeeded,
        queries_failed,
        total_raw_results,
        len(collected),
    )

    return collected, search_meta