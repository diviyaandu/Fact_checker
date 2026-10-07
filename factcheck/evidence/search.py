"""
Given a claim, retrieves candidate evidence from the open web via Tavily and
scores each source's credibility tier.

Also tracks whether the search came back with genuinely zero raw results
(as opposed to results that existed but were irrelevant) — this distinction
matters for claims that assert a specific named entity exists.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

from tavily import TavilyClient

from factcheck import config
from factcheck.evidence.credibility import score_credibility
from factcheck.logging_utils import logger
from factcheck.models import Source

tavily_client = TavilyClient(api_key=config.TAVILY_API_KEY)


def _fallback_queries(claim_text: str) -> list[str]:
    """
    Used only if a claim reaches this function with no queries attached.
    Normal path: queries are generated during claim extraction, so this
    function makes no LLM call of its own.
    """
    return [
        claim_text,
        f"{claim_text} fact check",
        f"is it true that {claim_text.lower()}",
    ]


def _extract_publisher(url: str) -> str:
    return urlparse(url).netloc.replace("www.", "")


def _search_tavily(query: str) -> tuple[str, list[dict]]:
    """Run one Tavily search and return the query plus raw results."""
    logger.info(
        "TAVILY call | query=%s | max_results=%d",
        query,
        config.MAX_RESULTS_PER_QUERY,
    )

    start = time.time()

    try:
        response = tavily_client.search(
            query=query,
            max_results=config.MAX_RESULTS_PER_QUERY,
        )

        results = response.get("results", [])

        logger.info(
            "TAVILY success | query=%s | results=%d | duration=%.2fs",
            query,
            len(results),
            time.time() - start,
        )

        return query, results

    except Exception:
        logger.exception("TAVILY failure | query=%s", query)
        return query, []


def _run_searches(queries: list[str]) -> tuple[dict[str, list[dict]], dict]:
    """Runs all queries concurrently (small worker pool, to stay within free-tier
    rate limits). Returns ({query: raw results}, counters)."""
    results_by_query: dict[str, list[dict]] = {}
    counters = {"total_raw_results": 0, "queries_succeeded": 0, "queries_failed": 0}

    with ThreadPoolExecutor(max_workers=config.MAX_CONCURRENT_SEARCHES) as executor:
        futures = {executor.submit(_search_tavily, query): query for query in queries}

        for future in as_completed(futures):
            query = futures[future]

            try:
                returned_query, results = future.result()
                results_by_query[returned_query] = results
                counters["queries_succeeded"] += 1
                counters["total_raw_results"] += len(results)

            except Exception:
                logger.exception("TAVILY worker failure | query=%s", query)
                counters["queries_failed"] += 1

    return results_by_query, counters


def _collect_sources(queries: list[str], results_by_query: dict[str, list[dict]]) -> list[Source]:
    """Turns raw results into Sources: one per publisher, capped, best credibility first.
    Processed in original query order so the output is deterministic."""
    seen_publishers: set[str] = set()
    collected: list[Source] = []

    for query in queries:
        for result in results_by_query.get(query, []):
            if len(collected) >= config.MAX_SOURCES_PER_CLAIM:
                break

            url = result.get("url", "")
            publisher = _extract_publisher(url)

            if publisher in seen_publishers:
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
                    relevant_text=content[: config.MAX_SNIPPET_CHARS],
                    credibility_tier=score_credibility(publisher),
                )
            )
            seen_publishers.add(publisher)

    collected.sort(key=lambda s: s.credibility_tier)
    return collected


def search_evidence_for_claim(
    claim_text: str,
    category: str,
    queries: list[str] | None = None,
) -> tuple[list[Source], dict]:
    """
    Returns (sources, search_meta).

    queries: the search queries generated during claim extraction.
    If omitted, cheap template queries are used instead.
    """
    queries = queries or _fallback_queries(claim_text)

    results_by_query, counters = _run_searches(queries)
    collected = _collect_sources(queries, results_by_query)

    search_meta = {
        "queries_used": queries,
        "total_raw_results": counters["total_raw_results"],
        "queries_succeeded": counters["queries_succeeded"],
        "queries_failed": counters["queries_failed"],
        "sources_found": len(collected),
    }

    logger.info(
        "SEARCH complete | queries=%d | succeeded=%d | failed=%d | raw_results=%d | sources=%d",
        len(queries),
        counters["queries_succeeded"],
        counters["queries_failed"],
        counters["total_raw_results"],
        len(collected),
    )

    return collected, search_meta
