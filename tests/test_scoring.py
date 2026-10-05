"""
Tests for the deterministic confidence scoring and its integration with the
batch fact-check + pipeline flow. No network, no API keys, no extra
dependencies:  python -m unittest discover -s tests -v
"""

import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

# Make the project root importable and provide dummy keys so the Groq/Tavily
# clients can be constructed at import time (nothing is ever called).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("TAVILY_API_KEY", "test-key")

# Everything that touches the network/GPU/UI is mocked below, so if a heavy SDK
# isn't installed (e.g. a bare CI box) fall back to an inert stand-in module
# just so the project's imports succeed. Installed packages are used as-is.
for _mod in ("groq", "tavily", "youtube_transcript_api", "yt_dlp", "faster_whisper", "streamlit"):
    try:
        __import__(_mod)
    except ImportError:
        _stub = type(sys)(_mod)
        _stub.__getattr__ = lambda name: (lambda *a, **k: SimpleNamespace())
        sys.modules[_mod] = _stub

import scoring  # noqa: E402
import fact_checker  # noqa: E402
import pipeline  # noqa: E402
import cache_utils  # noqa: E402
from claims import Claim  # noqa: E402
from search import Source  # noqa: E402


def src(publisher, tier, i=0):
    return Source(
        title=f"T{i}", url=f"https://{publisher}/{i}", publisher=publisher,
        relevant_text="snippet", credibility_tier=tier,
    )


def assess(idx, rel, stance, direct):
    return {"source_index": idx, "relevance": rel, "stance": stance, "directness": direct}


def conf(verdict, assessments, sources):
    return scoring.compute_confidence(
        verdict, scoring.parse_assessments(assessments, len(sources)), sources
    )


class ScoringFormula(unittest.TestCase):
    def test_unverifiable_is_zero(self):
        c, bd = conf("UNVERIFIABLE", [assess(0, 3, "supports", 3)], [src("nasa.gov", 1)])
        self.assertEqual(c, 0.0)

    def test_strong_corroborated_true_is_high_and_below_cap(self):
        sources = [src("nasa.gov", 1), src("nature.com", 2, 1), src("bbc.com", 3, 2)]
        a = [assess(i, 3, "supports", 3) for i in range(3)]
        c, bd = conf("TRUE", a, sources)
        self.assertGreater(c, 0.85)
        self.assertLessEqual(c, scoring.MAX_CONFIDENCE)
        self.assertFalse(bd["conflict_capped"])

    def test_formula_matches_hand_calculation(self):
        # One Tier-1 source, rel 3, directness 2, supports TRUE.
        c, bd = conf("TRUE", [assess(0, 3, "supports", 2)], [src("nasa.gov", 1)])
        expected = 0.20 * 1.0 + 0.25 * 1.0 + 0.35 * (1.0 * (2 / 3)) + 0.20 * (1 / 3)
        self.assertAlmostEqual(c, round(expected, 3), places=3)

    def test_low_credibility_lowers_confidence(self):
        hi, _ = conf("TRUE", [assess(0, 3, "supports", 3)], [src("nasa.gov", 1)])
        lo, _ = conf("TRUE", [assess(0, 3, "supports", 3)], [src("randomblog.xyz", 5)])
        self.assertGreater(hi, lo)

    def test_more_independent_sources_raise_confidence(self):
        one, _ = conf("TRUE", [assess(0, 3, "supports", 3)], [src("bbc.com", 3)])
        three, _ = conf(
            "TRUE", [assess(i, 3, "supports", 3) for i in range(3)],
            [src("bbc.com", 3), src("reuters.com", 3, 1), src("apnews.com", 3, 2)],
        )
        self.assertGreater(three, one)

    def test_same_registrable_domain_counts_once(self):
        a = src("news.bbc.co.uk", 3)
        b = src("www.bbc.co.uk", 3, 1)
        _, bd = conf("TRUE", [assess(0, 3, "supports", 3), assess(1, 3, "supports", 3)], [a, b])
        self.assertEqual(bd["independent_domains"], 1)

    def test_conflict_caps_confidence(self):
        sources = [src("nasa.gov", 1), src("nature.com", 2, 1), src("who.int", 1, 2), src("cdc.gov", 1, 3)]
        a = [assess(0, 3, "supports", 3), assess(1, 3, "supports", 3), assess(2, 3, "supports", 3),
             assess(3, 3, "contradicts", 3)]
        c, bd = conf("TRUE", a, sources)
        self.assertTrue(bd["conflict_capped"])
        self.assertLessEqual(c, scoring.CONFLICT_CAP)

        clean, _ = conf("TRUE", a[:3], sources[:3])
        self.assertGreater(clean, c)

    def test_true_with_only_contradicting_sources_has_no_aligned_evidence(self):
        c, bd = conf("TRUE", [assess(0, 3, "contradicts", 3)], [src("nasa.gov", 1)])
        self.assertEqual(c, 0.0)
        self.assertTrue(bd.get("no_aligned_evidence"))

    def test_neutral_irrelevant_or_missing_assessments_are_ignored(self):
        sources = [src("nasa.gov", 1), src("bbc.com", 3, 1)]
        a = [assess(0, 0, "supports", 3), assess(1, 3, "neutral", 3)]
        c, bd = conf("TRUE", a, sources)
        self.assertEqual(c, 0.0)
        c2, bd2 = conf("TRUE", [], sources)
        self.assertTrue(bd2.get("no_aligned_evidence"))

    def test_mixed_verdict_prefers_partial_evidence(self):
        sources = [src("nasa.gov", 1), src("nature.com", 2, 1)]
        partial, _ = conf("PARTIALLY TRUE", [assess(0, 3, "partial", 3), assess(1, 3, "partial", 3)], sources)
        all_support, _ = conf("PARTIALLY TRUE", [assess(0, 3, "supports", 3), assess(1, 3, "supports", 3)], sources)
        self.assertGreater(partial, all_support)

    def test_deterministic(self):
        sources = [src("nasa.gov", 1), src("bbc.com", 3, 1)]
        a = [assess(0, 3, "supports", 3), assess(1, 2, "supports", 2)]
        self.assertEqual(conf("TRUE", a, sources), conf("TRUE", a, sources))

    def test_parse_assessments_sanitises_bad_input(self):
        raw = [
            assess(0, 9, "SUPPORTS", -4),                 # clamped, case-insensitive
            assess(0, 3, "contradicts", 3),               # duplicate index dropped
            assess(7, 3, "supports", 3),                  # out of range
            {"source_index": "1", "relevance": 3},        # non-int index
            assess(1, "x", "banana", None),               # bad ratings / stance
            "garbage",
        ]
        parsed = scoring.parse_assessments(raw, 2)
        self.assertEqual([p.source_index for p in parsed], [0, 1])
        self.assertEqual((parsed[0].relevance, parsed[0].stance, parsed[0].directness), (3, "supports", 0))
        self.assertEqual((parsed[1].relevance, parsed[1].stance, parsed[1].directness), (0, "neutral", 0))
        self.assertEqual(scoring.parse_assessments("nope", 2), [])


class FactCheckIntegration(unittest.TestCase):
    def _run_batch(self, llm_payload, items):
        with mock.patch.object(fact_checker, "_call_llm_batch", return_value=json.dumps(llm_payload)):
            return fact_checker.fact_check_claims_batch(items)

    def test_llm_confidence_is_ignored(self):
        sources = [src("nasa.gov", 1)]
        payload = [{
            "claim_index": 0, "verdict": "TRUE", "confidence": 0.99,
            "source_assessments": [assess(0, 3, "supports", 3)],
            "summary": "s", "explanation": "e",
            "cited_sources": [{"publisher": "nasa.gov", "url": sources[0].url, "title": "T0", "relevant_text": "x"}],
        }]
        (v,) = self._run_batch(payload, [("claim", sources)])
        self.assertEqual(v.verdict, "TRUE")
        self.assertNotEqual(v.confidence, 0.99)
        self.assertAlmostEqual(v.confidence, scoring.compute_confidence(
            "TRUE", scoring.parse_assessments(payload[0]["source_assessments"], 1), sources)[0])
        self.assertIn("components", v.confidence_breakdown)

    def test_verdict_without_valid_citation_is_unverifiable_zero(self):
        sources = [src("nasa.gov", 1)]
        payload = [{"claim_index": 0, "verdict": "TRUE", "source_assessments": [assess(0, 3, "supports", 3)],
                    "summary": "s", "explanation": "e",
                    "cited_sources": [{"publisher": "x", "url": "https://fake/url", "title": "t", "relevant_text": "x"}]}]
        (v,) = self._run_batch(payload, [("claim", sources)])
        self.assertEqual((v.verdict, v.confidence), ("UNVERIFIABLE", 0.0))

    def test_missing_assessments_downgrade_to_unverifiable(self):
        sources = [src("nasa.gov", 1)]
        payload = [{"claim_index": 0, "verdict": "FALSE", "summary": "s", "explanation": "e",
                    "cited_sources": [{"publisher": "nasa.gov", "url": sources[0].url, "title": "t", "relevant_text": "x"}]}]
        (v,) = self._run_batch(payload, [("claim", sources)])
        self.assertEqual((v.verdict, v.confidence), ("UNVERIFIABLE", 0.0))

    def test_unverifiable_from_llm_is_zero(self):
        payload = [{"claim_index": 0, "verdict": "UNVERIFIABLE", "confidence": 0.8,
                    "summary": "s", "explanation": "e", "cited_sources": []}]
        (v,) = self._run_batch(payload, [("claim", [src("nasa.gov", 1)])])
        self.assertEqual(v.confidence, 0.0)

    def test_no_sources_paths_unchanged(self):
        v = fact_checker.fact_check_no_sources({"total_raw_results": 0, "queries_succeeded": 3})
        self.assertEqual((v.verdict, v.confidence), ("FALSE", 0.45))
        v = fact_checker.fact_check_no_sources({"queries_succeeded": 0})
        self.assertEqual((v.verdict, v.confidence), ("UNVERIFIABLE", 0.0))

    def test_old_cached_verdict_still_loads(self):
        old = {"verdict": "TRUE", "confidence": 0.8, "summary": "s", "explanation": "e", "cited_sources": []}
        self.assertIsNone(fact_checker.Verdict(**old).confidence_breakdown)


class PipelineSmoke(unittest.TestCase):
    """Runs pipeline.get_claim_verdicts end to end with search + Groq mocked
    and the cache redirected to a temp file."""

    def test_pipeline_end_to_end(self):
        claims = [
            Claim(claim_id=i, claim_text=f"claim {i}", start_time="0:00", end_time="0:10",
                  category="Science", importance="high", search_queries=["q1", "q2", "q3"])
            for i in range(3)
        ]
        sources = [src("nasa.gov", 1), src("bbc.com", 3, 1)]

        def fake_search(text, category, queries):
            if text == "claim 2":   # zero results anywhere -> local heuristic path
                return [], {"total_raw_results": 0, "queries_succeeded": 3, "queries_used": queries}
            return sources, {"total_raw_results": 6, "queries_succeeded": 3, "queries_used": queries}

        def fake_llm(prompt, batch_size):
            return json.dumps([{
                "claim_index": i, "verdict": "TRUE", "confidence": 0.5,
                "source_assessments": [assess(0, 3, "supports", 3), assess(1, 3, "supports", 2)],
                "summary": "s", "explanation": "e",
                "cited_sources": [{"publisher": "nasa.gov", "url": sources[0].url, "title": "T0", "relevant_text": "x"}],
            } for i in range(batch_size)])

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(cache_utils, "_CACHE_PATH", os.path.join(tmp, "c.json")), \
                mock.patch.object(pipeline, "search_evidence_for_claim", side_effect=fake_search), \
                mock.patch.object(fact_checker, "_call_llm_batch", side_effect=fake_llm) as llm, \
                self.assertLogs("fact_checker", level="INFO") as logs:
            results = pipeline.get_claim_verdicts(claims, use_cache=True)

            # 2 claims with sources, BATCH_SIZE=2 -> exactly one Groq call.
            self.assertEqual(llm.call_count, 1)
            self.assertEqual(results[0]["verdict"].verdict, "TRUE")
            self.assertTrue(0 < results[0]["verdict"].confidence <= scoring.MAX_CONFIDENCE)
            self.assertEqual(results[2]["verdict"].confidence, 0.45)

            # Component scores are inspectable in the logs.
            self.assertTrue(any("CONFIDENCE | claim_id=0 | relevance=" in m for m in logs.output))

            # Second run is served from cache, breakdown preserved.
            again = pipeline.get_claim_verdicts(claims, use_cache=True)
            self.assertTrue(again[0]["from_cache"])
            self.assertEqual(again[0]["verdict"].confidence, results[0]["verdict"].confidence)
            self.assertIsNotNone(again[0]["verdict"].confidence_breakdown)


if __name__ == "__main__":
    unittest.main()
