"""NLI second opinion (model mocked; no torch/transformers needed):
python -m unittest discover -s tests -v"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install()

from factcheck import config, scoring  # noqa: E402
from factcheck.models import Source  # noqa: E402
from factcheck.scoring import nli  # noqa: E402

ENT = nli.result_from_probs({"entailment": 0.92, "neutral": 0.05, "contradiction": 0.03})
CON = nli.result_from_probs({"entailment": 0.03, "neutral": 0.05, "contradiction": 0.92})
NEU = nli.result_from_probs({"entailment": 0.08, "neutral": 0.84, "contradiction": 0.08})


def src(publisher, tier, i=0):
    return Source(title=f"T{i}", url=f"https://{publisher}/{i}", publisher=publisher,
                  relevant_text=f"snippet {i}", credibility_tier=tier)


def conf(verdict, stance, results, n=1):
    sources = [src(f"d{i}.org", 2, i) for i in range(n)]
    a = [{"source_index": i, "relevance": 3, "stance": stance, "directness": 3} for i in range(n)]
    return scoring.compute_confidence(
        verdict, scoring.parse_assessments(a, n), sources,
        {i: results for i in range(n)} if results else None,
    )


class Labels(unittest.TestCase):
    def test_predicted_labels(self):
        self.assertEqual(ENT.label, "entailment")
        self.assertEqual(CON.label, "contradiction")
        self.assertEqual(NEU.label, "neutral")

    def test_stance_mapping(self):
        m = nli.stance_to_nli_label
        self.assertEqual((m("supports"), m("contradicts"), m("partial"), m("neutral")),
                         ("entailment", "contradiction", "neutral", "neutral"))

    def test_classify_pairs_uses_predictor_and_disabled_returns_empty(self):
        raw = [{"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8}]
        with mock.patch.object(nli, "_predict", return_value=raw), \
                mock.patch.object(config, "NLI_ENABLED", True):
            (r,) = nli.classify_pairs([("snippet", "claim")])
        self.assertEqual(r.label, "contradiction")
        self.assertEqual(nli.classify_pairs([("s", "c")]), [])   # disabled in tests

    def test_unavailable_model_degrades_to_empty(self):
        with mock.patch.object(nli, "_predict", side_effect=ImportError("no torch")), \
                mock.patch.object(config, "NLI_ENABLED", True):
            self.assertEqual(nli.classify_pairs([("s", "c")]), [])

    def test_nli_for_claims_single_batched_call(self):
        sources = [src("a.org", 1, 0), src("b.org", 1, 1)]
        raw = [ENT.probs, CON.probs, NEU.probs]
        with mock.patch.object(nli, "_predict", return_value=raw) as p, \
                mock.patch.object(config, "NLI_ENABLED", True):
            out = nli.nli_for_claims([("c1", sources, [0, 1]), ("c2", sources, [1])])
        self.assertEqual(p.call_count, 1)
        self.assertEqual([out[0][0].label, out[0][1].label, out[1][1].label],
                         ["entailment", "contradiction", "neutral"])


class Agreement(unittest.TestCase):
    def test_agreement_values(self):
        self.assertGreater(nli.source_agreement("supports", ENT), 0.8)
        self.assertGreater(nli.source_agreement("contradicts", CON), 0.8)
        self.assertLess(nli.source_agreement("supports", CON), -0.8)
        self.assertEqual(nli.source_agreement("supports", NEU), 0.0)
        self.assertEqual(nli.source_agreement("partial", ENT), 0.0)


class ConfidenceEffect(unittest.TestCase):
    def test_agreement_raises_confidence(self):
        base, _ = conf("TRUE", "supports", None)
        c, bd = conf("TRUE", "supports", ENT)
        self.assertGreater(c, base)
        self.assertAlmostEqual(c - base, scoring.constants.NLI_MAX_BOOST * bd["nli"]["agreement"], places=2)

    def test_agreement_raises_confidence_for_false_verdict(self):
        base, _ = conf("FALSE", "contradicts", None)
        self.assertGreater(conf("FALSE", "contradicts", CON)[0], base)

    def test_disagreement_lowers_confidence(self):
        base, _ = conf("TRUE", "supports", None)
        c, bd = conf("TRUE", "supports", CON)
        self.assertLess(c, base)
        self.assertLess(bd["nli"]["adjustment"], 0)
        self.assertFalse(bd["per_source"][0]["nli"]["agrees"])

    def test_neutral_nli_gives_no_adjustment(self):
        base, _ = conf("TRUE", "supports", None)
        c, bd = conf("TRUE", "supports", NEU)
        self.assertEqual(c, base)
        self.assertEqual(bd["nli"]["adjustment"], 0.0)

    def test_penalty_larger_than_boost(self):
        base, _ = conf("TRUE", "supports", None)
        self.assertGreater(base - conf("TRUE", "supports", CON)[0],
                           conf("TRUE", "supports", ENT)[0] - base)

    def test_bounded_and_cap_respected(self):
        for stance_verdict in (("supports", "TRUE"), ("contradicts", "FALSE"), ("partial", "PARTIALLY TRUE")):
            for r in (ENT, CON, NEU):
                c, _ = conf(stance_verdict[1], stance_verdict[0], r, n=4)
                self.assertTrue(0.0 <= c <= scoring.MAX_CONFIDENCE)
        # conflict cap still binds with full NLI agreement
        sources = [src("a.org", 3, 0), src("b.org", 3, 1), src("c.org", 3, 2), src("d.org", 1, 3)]
        a = [{"source_index": i, "relevance": 3, "directness": 3,
              "stance": "contradicts" if i == 3 else "supports"} for i in range(4)]
        c, bd = scoring.compute_confidence("TRUE", scoring.parse_assessments(a, 4), sources,
                                           {i: ENT for i in range(4)})
        self.assertTrue(bd["conflict_capped"])
        self.assertLessEqual(c, scoring.CONFLICT_CAP)

    def test_no_nli_is_identical_to_before_and_no_breakdown_key(self):
        c, bd = conf("TRUE", "supports", None)
        self.assertNotIn("nli", bd)
        self.assertEqual(c, conf("TRUE", "supports", {})[0])

    def test_no_nli_cannot_rescue_missing_evidence(self):
        c, bd = conf("TRUE", "contradicts", ENT)
        self.assertEqual(c, 0.0)
        self.assertTrue(bd.get("no_aligned_evidence"))


class BatchIntegration(unittest.TestCase):
    def _run(self, nli_result):
        import json
        from factcheck.verdicts import checker, fact_check_claims_batch
        sources = [src("nasa.gov", 1)]
        payload = [{"claim_index": 0, "verdict": "TRUE", "summary": "s", "explanation": "e",
                    "source_assessments": [{"source_index": 0, "relevance": 3, "stance": "supports", "directness": 3}],
                    "cited_sources": [{"publisher": "nasa.gov", "url": sources[0].url, "title": "t", "relevant_text": "x"}]}]
        with mock.patch.object(checker, "_call_llm_batch", return_value=json.dumps(payload)), \
                mock.patch.object(checker, "nli_for_claims", return_value=[{0: nli_result} if nli_result else {}]):
            return fact_check_claims_batch([("claim", sources)])[0]

    def test_groq_nli_agreement_and_disagreement(self):
        none, agree, disagree = self._run(None), self._run(ENT), self._run(CON)
        self.assertGreater(agree.confidence, none.confidence)
        self.assertLess(disagree.confidence, none.confidence)
        self.assertEqual(agree.verdict, "TRUE")
        self.assertIn("nli", agree.confidence_breakdown)
        self.assertEqual(agree.confidence_breakdown["per_source"][0]["nli"]["label"], "entailment")


if __name__ == "__main__":
    unittest.main()
