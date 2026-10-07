"""Verdict plumbing (LLM mocked): prompt layout, JSON handling, batching, citations."""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install()

from factcheck.models import Source  # noqa: E402
from factcheck.verdicts import BATCH_SIZE, checker, fact_check_claims_batch  # noqa: E402
from factcheck.verdicts.parsing import (  # noqa: E402
    match_objects_to_items,
    parse_batch_response,
    validate_and_verify,
)
from factcheck.verdicts.prompt import SYSTEM_PROMPT, build_batch_user_prompt  # noqa: E402


def src(i=0, tier=1):
    return Source(title=f"Title {i}", url=f"https://site{i}.org/p", publisher=f"site{i}.org",
                  relevant_text=f"snippet {i}", credibility_tier=tier)


def good(i, sources, claim_index=0, verdict="TRUE"):
    return {
        "claim_index": claim_index, "verdict": verdict, "summary": "s", "explanation": "e",
        "source_assessments": [{"source_index": 0, "relevance": 3, "stance": "supports", "directness": 3}],
        "cited_sources": [{"publisher": sources[0].publisher, "url": sources[0].url,
                           "title": sources[0].title, "relevant_text": "x"}],
    }


class Prompt(unittest.TestCase):
    def test_user_prompt_numbers_claims_and_sources(self):
        text = build_batch_user_prompt([("first claim", [src(0), src(1, tier=3)]), ("second claim", [src(2)])])
        self.assertIn("=== CLAIM 0 ===", text)
        self.assertIn("=== CLAIM 1 ===", text)
        self.assertIn("[1] Publisher: site1.org (Credibility Tier: 3)", text)
        self.assertIn("Snippet: snippet 2", text)
        self.assertLess(text.index("first claim"), text.index("second claim"))

    def test_system_prompt_asks_for_assessments_not_a_confidence_number(self):
        self.assertIn("source_assessments", SYSTEM_PROMPT)
        self.assertIn("Do NOT output any confidence", SYSTEM_PROMPT)
        self.assertNotIn('"confidence"', SYSTEM_PROMPT)


class Parsing(unittest.TestCase):
    def test_parse_batch_response(self):
        self.assertEqual(parse_batch_response("[1, 2]"), [1, 2])
        self.assertIsNone(parse_batch_response("{}"))
        self.assertIsNone(parse_batch_response("not json"))

    def test_match_by_claim_index_with_positional_fallback(self):
        a, b, c = {"claim_index": 1, "n": "a"}, {"n": "b"}, {"claim_index": 1, "n": "dup"}
        matched = match_objects_to_items([a, b, c, "junk"], 3)
        self.assertEqual(matched[1]["n"], "a")          # indexed object lands in its slot
        self.assertEqual(matched[0]["n"], "b")          # unindexed ones fill gaps in order
        self.assertEqual(matched[2]["n"], "dup")
        self.assertEqual(match_objects_to_items([], 2), [None, None])

    def test_only_real_urls_survive_as_citations(self):
        sources = [src(0)]
        obj = good(0, sources)
        obj["cited_sources"].append({"publisher": "x", "url": "https://made-up.example", "title": "t",
                                     "relevant_text": "x"})
        v = validate_and_verify(obj, sources)
        self.assertEqual([c.url for c in v.cited_sources], [sources[0].url])

    def test_invalid_verdict_label_falls_back_safely(self):
        v = validate_and_verify({"verdict": "MAYBE", "cited_sources": []}, [src(0)])
        self.assertEqual((v.verdict, v.confidence), ("UNVERIFIABLE", 0.0))

    def test_stance_must_match_verdict(self):
        sources = [src(0)]
        obj = good(0, sources, verdict="FALSE")      # assessments say "supports"
        v = validate_and_verify(obj, sources)
        self.assertEqual((v.verdict, v.confidence), ("UNVERIFIABLE", 0.0))
        self.assertEqual(v.confidence_breakdown["reason"], "no_aligned_evidence")

    def test_confidence_comes_from_the_formula_not_the_model(self):
        sources = [src(0)]
        obj = good(0, sources)
        obj["confidence"] = 0.99
        v = validate_and_verify(obj, sources)
        self.assertNotEqual(v.confidence, 0.99)
        self.assertIn("components", v.confidence_breakdown)


class Batching(unittest.TestCase):
    def call(self, replies, n_items=2):
        sources = [[src(i)] for i in range(n_items)]
        items = [(f"claim {i}", sources[i]) for i in range(n_items)]
        with mock.patch.object(checker, "_call_llm_batch", side_effect=replies) as llm:
            return fact_check_claims_batch(items), llm, sources

    def test_batch_size_default(self):
        self.assertEqual(BATCH_SIZE, 2)

    def test_empty_input_makes_no_call(self):
        with mock.patch.object(checker, "_call_llm_batch") as llm:
            self.assertEqual(fact_check_claims_batch([]), [])
        llm.assert_not_called()

    def test_results_follow_input_order_even_if_model_reorders(self):
        sources = [[src(0)], [src(1)]]
        reply = json.dumps([good(1, sources[1], claim_index=1, verdict="TRUE"),
                            good(0, sources[0], claim_index=0, verdict="TRUE")])
        verdicts, llm, _ = self.call([reply])
        self.assertEqual(llm.call_count, 1)             # both claims in ONE Groq call
        self.assertEqual([v.cited_sources[0].publisher for v in verdicts], ["site0.org", "site1.org"])

    def test_bad_json_gets_one_stricter_retry(self):
        sources = [[src(0)]]
        ok = json.dumps([good(0, sources[0])])
        verdicts, llm, _ = self.call(["garbage", ok], n_items=1)
        self.assertEqual(llm.call_count, 2)
        self.assertIn("REMINDER", llm.call_args_list[1].args[0])
        self.assertEqual(verdicts[0].verdict, "TRUE")

    def test_two_bad_replies_give_fallback_verdicts(self):
        verdicts, llm, _ = self.call(["garbage", "more garbage"])
        self.assertEqual(llm.call_count, 2)
        self.assertEqual([(v.verdict, v.confidence) for v in verdicts], [("UNVERIFIABLE", 0.0)] * 2)

    def test_missing_result_for_one_claim(self):
        sources = [[src(0)], [src(1)]]
        verdicts, _, _ = self.call([json.dumps([good(0, sources[0], claim_index=0)])])
        self.assertEqual(verdicts[0].verdict, "TRUE")
        self.assertEqual(verdicts[1].verdict, "UNVERIFIABLE")

    def test_code_fenced_reply_is_accepted(self):
        sources = [[src(0)]]
        reply = "```json\n" + json.dumps([good(0, sources[0])]) + "\n```"
        verdicts, llm, _ = self.call([reply], n_items=1)
        self.assertEqual(llm.call_count, 1)
        self.assertEqual(verdicts[0].verdict, "TRUE")

    def test_groq_call_uses_the_verdict_model_and_scaled_token_budget(self):
        from factcheck import config
        with mock.patch.object(checker.llm, "chat", return_value="[]") as chat:
            checker._call_llm_batch("prompt", 2)
        kwargs = chat.call_args.kwargs
        self.assertEqual(kwargs["model"], config.VERDICT_MODEL)
        self.assertEqual(kwargs["stage"], "fact_check_batch")
        self.assertEqual(kwargs["max_tokens"], 3200)
        self.assertEqual(kwargs["log_fields"], " | batch_size=2")


if __name__ == "__main__":
    unittest.main()
