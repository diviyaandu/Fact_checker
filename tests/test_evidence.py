"""Evidence search (Tavily mocked): de-duplication, caps, ranking, metadata."""

import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install()

from factcheck.evidence import search as search_mod  # noqa: E402
from factcheck.evidence import label_for_tier, score_credibility  # noqa: E402
from factcheck.evidence.search import search_evidence_for_claim  # noqa: E402


def hit(url, content="some evidence text", title="T"):
    return {"url": url, "content": content, "title": title}


def run(results_by_query, queries=None, delay=None):
    """Runs the search with _search_tavily faked: {query: [raw hits]}."""
    def fake(query):
        if delay:
            time.sleep(delay.get(query, 0))
        return query, results_by_query.get(query, [])

    with mock.patch.object(search_mod, "_search_tavily", side_effect=fake):
        return search_evidence_for_claim("claim text", "Science", queries or list(results_by_query))


class Credibility(unittest.TestCase):
    def test_tiers(self):
        self.assertEqual(score_credibility("nasa.gov"), 1)
        self.assertEqual(score_credibility("nature.com"), 2)
        self.assertEqual(score_credibility("bbc.co.uk"), 3)
        self.assertEqual(score_credibility("en.wikipedia.org"), 4)
        self.assertEqual(score_credibility("randomblog.xyz"), 5)
        self.assertIn("Tier 1", label_for_tier(1))
        self.assertIn("Unrated", label_for_tier(99))


class SearchEvidence(unittest.TestCase):
    def test_one_source_per_publisher_and_www_stripped(self):
        sources, meta = run({
            "q1": [hit("https://www.nasa.gov/a"), hit("https://nasa.gov/b")],
            "q2": [hit("https://www.bbc.com/c")],
        })
        self.assertEqual([s.publisher for s in sources], ["nasa.gov", "bbc.com"])
        self.assertEqual(meta["sources_found"], 2)

    def test_sorted_best_credibility_first(self):
        sources, _ = run({"q": [hit("https://randomblog.xyz/1"), hit("https://en.wikipedia.org/2"),
                                hit("https://nasa.gov/3")]})
        self.assertEqual([s.credibility_tier for s in sources], [1, 4, 5])

    def test_capped_at_five_sources(self):
        sources, meta = run({"q": [hit(f"https://site{i}.com/x") for i in range(9)]})
        self.assertEqual(len(sources), 5)
        self.assertEqual(meta["total_raw_results"], 9)

    def test_empty_content_skipped_and_snippet_truncated(self):
        sources, _ = run({"q": [hit("https://a.com/1", content="   "),
                                hit("https://b.com/2", content="x" * 2000)]})
        self.assertEqual([s.publisher for s in sources], ["b.com"])
        self.assertEqual(len(sources[0].relevant_text), 800)

    def test_missing_fields_get_defaults(self):
        sources, _ = run({"q": [{"url": "https://a.com/1", "content": "text"}]})
        self.assertEqual(sources[0].title, "Untitled")
        self.assertIsNone(sources[0].published_date)

    def test_output_order_follows_query_order_not_completion_order(self):
        # q1 finishes last, but its hit must still win the tie for the same tier.
        sources, _ = run(
            {"q1": [hit("https://first.com/1")], "q2": [hit("https://second.com/2")]},
            delay={"q1": 0.15},
        )
        self.assertEqual([s.publisher for s in sources], ["first.com", "second.com"])

    def test_metadata_for_a_normal_search(self):
        _, meta = run({"q1": [hit("https://a.com/1")], "q2": [], "q3": [hit("https://b.com/2")]})
        self.assertEqual(meta["queries_used"], ["q1", "q2", "q3"])
        self.assertEqual(meta["queries_succeeded"], 3)   # an empty result still counts as a success
        self.assertEqual(meta["queries_failed"], 0)
        self.assertEqual(meta["total_raw_results"], 2)

    def test_zero_results_everywhere(self):
        sources, meta = run({"q1": [], "q2": []})
        self.assertEqual(sources, [])
        self.assertEqual((meta["total_raw_results"], meta["queries_succeeded"]), (0, 2))

    def test_fallback_queries_when_none_given(self):
        seen = []

        def fake(query):
            seen.append(query)
            return query, []

        with mock.patch.object(search_mod, "_search_tavily", side_effect=fake):
            _, meta = search_evidence_for_claim("Water boils", "Science", None)
        self.assertEqual(sorted(seen), sorted(meta["queries_used"]))
        self.assertEqual(len(seen), 3)
        self.assertIn("is it true that water boils", seen)

    def test_worker_exception_counts_as_failed_query(self):
        def fake(query):
            if query == "bad":
                raise RuntimeError("boom")
            return query, [hit("https://ok.com/1")]

        with mock.patch.object(search_mod, "_search_tavily", side_effect=fake):
            sources, meta = search_evidence_for_claim("c", "Science", ["good", "bad"])
        self.assertEqual((meta["queries_succeeded"], meta["queries_failed"]), (1, 1))
        self.assertEqual(len(sources), 1)

    def test_tavily_error_is_swallowed_per_query(self):
        client = mock.MagicMock()
        client.search.side_effect = RuntimeError("rate limited")
        with mock.patch.object(search_mod, "tavily_client", client):
            query, results = search_mod._search_tavily("anything")
        self.assertEqual((query, results), ("anything", []))

    def test_tavily_called_with_configured_limit(self):
        client = mock.MagicMock()
        client.search.return_value = {"results": [hit("https://a.com/1")]}
        with mock.patch.object(search_mod, "tavily_client", client):
            _, results = search_mod._search_tavily("q")
        client.search.assert_called_once_with(query="q", max_results=3)
        self.assertEqual(len(results), 1)

    def test_searches_run_concurrently_but_within_the_worker_limit(self):
        lock, state = threading.Lock(), {"now": 0, "peak": 0}

        def fake(query):
            with lock:
                state["now"] += 1
                state["peak"] = max(state["peak"], state["now"])
            time.sleep(0.05)
            with lock:
                state["now"] -= 1
            return query, []

        with mock.patch.object(search_mod, "_search_tavily", side_effect=fake):
            search_evidence_for_claim("c", "Science", ["a", "b", "c", "d"])
        self.assertEqual(state["peak"], 2)   # MAX_CONCURRENT_SEARCHES


if __name__ == "__main__":
    unittest.main()
