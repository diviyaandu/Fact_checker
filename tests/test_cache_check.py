"""pipeline.is_fully_cached against the real cache module (temp file, no network)."""

import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install()

from factcheck import cache  # noqa: E402
from factcheck.pipeline import cache_check, is_fully_cached  # noqa: E402


def claim_dict(i, text):
    return {"claim_id": i, "claim_text": text, "start_time": "0:00", "end_time": "0:10",
            "category": "Science", "importance": "high", "search_queries": []}


class IsFullyCached(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(cache, "_CACHE_PATH", os.path.join(self.tmp.name, "c.json"))
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def fill(self, transcript=True, claims=("a", "b"), results=("a", "b")):
        if transcript:
            cache.set("transcript", "vid", {"segments": [{"text": "x"}], "method": "m"})
        if claims is not None:
            cache.set("claims", "vid", [claim_dict(i, t) for i, t in enumerate(claims)])
        for t in results:
            cache.set("claim_results", cache_check.claim_cache_key(t), {"verdict": {}})

    def test_everything_cached(self):
        self.fill()
        self.assertTrue(is_fully_cached("vid", True))

    def test_cache_disabled(self):
        self.fill()
        self.assertFalse(is_fully_cached("vid", False))

    def test_missing_transcript(self):
        self.fill(transcript=False)
        self.assertFalse(is_fully_cached("vid", True))

    def test_missing_claims(self):
        self.fill(claims=None, results=())
        self.assertFalse(is_fully_cached("vid", True))

    def test_one_claim_result_missing(self):
        self.fill(results=("a",))
        self.assertFalse(is_fully_cached("vid", True))

    def test_cached_video_with_no_claims(self):
        self.fill(claims=(), results=())
        self.assertTrue(is_fully_cached("vid", True))

    def test_lookup_key_matches_get_claim_verdicts(self):
        # The check and the real lookup must share one key function.
        self.assertEqual(cache_check.claim_cache_key("a"), cache.make_key("claim", "a"))

    def test_is_silent(self):
        self.fill()
        with self.assertNoLogs("fact_checker", level="INFO"):
            is_fully_cached("vid", True)


if __name__ == "__main__":
    unittest.main()
