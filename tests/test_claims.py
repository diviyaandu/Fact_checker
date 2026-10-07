"""Claim extraction (LLM mocked): validation, timestamps, caps, retries, fallbacks."""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install()

from factcheck.claims import extractor, extract_claims  # noqa: E402


def segments(n=6):
    return [{"start": i * 5, "end": i * 5 + 5, "start_str": f"00:{i * 5:02d}",
             "end_str": f"00:{i * 5 + 5:02d}", "text": f"sentence {i}"} for i in range(n)]


def cand(text="The sky is blue", idx=(1,), category="Science", **extra):
    return {"claim_text": text, "segment_indices": list(idx), "category": category,
            "importance": "high", "search_queries": ["q1", "q2", "q3"], **extra}


def run(outputs, segs=None):
    """Runs extract_claims with the LLM returning `outputs` in turn."""
    with mock.patch.object(extractor, "_call_llm", side_effect=outputs) as llm:
        claims = extract_claims(segs if segs is not None else segments())
    return claims, llm


class ExtractClaims(unittest.TestCase):
    def test_no_segments_means_no_llm_call(self):
        claims, llm = run([], segs=[])
        self.assertEqual(claims, [])
        llm.assert_not_called()

    def test_timestamps_come_from_the_transcript(self):
        claims, _ = run([json.dumps([cand(idx=(1, 3))])])
        self.assertEqual(len(claims), 1)
        self.assertEqual((claims[0].start_time, claims[0].end_time), ("00:05", "00:20"))
        self.assertEqual(claims[0].claim_id, 1)
        self.assertEqual(claims[0].search_queries, ["q1", "q2", "q3"])

    def test_code_fences_are_stripped(self):
        claims, _ = run(["```json\n" + json.dumps([cand()]) + "\n```"])
        self.assertEqual(len(claims), 1)

    def test_rejects_bad_candidates_and_keeps_ids_contiguous(self):
        bad = [
            cand(idx=(99,)),                   # segment doesn't exist
            cand(idx=(-1,)),                   # negative index
            cand(idx=()),                      # no indices
            cand(category="Sports"),           # unknown category
            cand(text="   "),                  # empty claim text
            "not a dict",
            cand(idx=("1",)),                  # indices must be ints
        ]
        good = cand("kept one", idx=(2,))
        claims, _ = run([json.dumps(bad[:3] + [good] + bad[3:])])
        self.assertEqual([c.claim_text for c in claims], ["kept one"])
        self.assertEqual([c.claim_id for c in claims], [1])

    def test_hard_cap_of_five(self):
        claims, _ = run([json.dumps([cand(f"claim {i}") for i in range(8)])])
        self.assertEqual(len(claims), 5)
        self.assertEqual([c.claim_id for c in claims], [1, 2, 3, 4, 5])

    def test_cap_applies_before_validation(self):
        # Only the first five candidates are even considered.
        junk = [cand(idx=(99,)) for _ in range(5)]
        claims, _ = run([json.dumps(junk + [cand("sixth")])])
        self.assertEqual(claims, [])

    def test_missing_queries_get_template_fallbacks(self):
        c = cand("Water boils at 100C")
        del c["search_queries"]
        (claim,), _ = run([json.dumps([c])])
        self.assertEqual(claim.search_queries, [
            "Water boils at 100C",
            "Water boils at 100C fact check",
            "is it true that water boils at 100c",
        ])

    def test_blank_or_malformed_queries_fall_back_too(self):
        for bad in ("a string", ["ok", ""], [1, 2], []):
            (claim,), _ = run([json.dumps([cand(search_queries=bad)])])
            self.assertEqual(len(claim.search_queries), 3, bad)
            self.assertTrue(claim.search_queries[1].endswith("fact check"), bad)

    def test_extra_queries_are_trimmed_to_three(self):
        (claim,), _ = run([json.dumps([cand(search_queries=["a", "b", "c", "d", "e"])])])
        self.assertEqual(claim.search_queries, ["a", "b", "c"])

    def test_importance_defaults_to_medium(self):
        c = cand()
        del c["importance"]
        (claim,), _ = run([json.dumps([c])])
        self.assertEqual(claim.importance, "medium")

    def test_bad_json_retries_once_with_a_reminder(self):
        claims, llm = run(["not json", json.dumps([cand()])])
        self.assertEqual(len(claims), 1)
        self.assertEqual(llm.call_count, 2)
        self.assertIn("REMINDER", llm.call_args_list[1].args[0])

    def test_gives_up_cleanly_after_two_bad_replies(self):
        claims, llm = run(["nope", "still nope"])
        self.assertEqual(claims, [])
        self.assertEqual(llm.call_count, 2)

    def test_non_list_json_gives_no_claims(self):
        claims, llm = run([json.dumps({"claims": []})])
        self.assertEqual(claims, [])
        self.assertEqual(llm.call_count, 1)

    def test_transcript_is_numbered_for_the_model(self):
        _, llm = run([json.dumps([])])
        sent = llm.call_args.args[0]
        self.assertTrue(sent.startswith("[0] sentence 0\n[1] sentence 1"))


if __name__ == "__main__":
    unittest.main()
