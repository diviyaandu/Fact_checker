"""Evaluation metrics + runner flow. No network or API keys (pipeline is faked)."""

import csv
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import _stubs

_stubs.install()

from evaluation import run_eval
from evaluation.metrics import LABELS, compute_metrics, confusion_matrix


class MetricsTests(unittest.TestCase):
    def test_hand_checked_values(self):
        t = ["TRUE", "TRUE", "FALSE", "FALSE", "UNVERIFIABLE"]
        p = ["TRUE", "FALSE", "FALSE", "FALSE", "FALSE"]
        m = compute_metrics(t, p)

        self.assertAlmostEqual(m["per_class"]["TRUE"]["precision"], 1.0)
        self.assertAlmostEqual(m["per_class"]["TRUE"]["recall"], 0.5)
        self.assertAlmostEqual(m["per_class"]["TRUE"]["f1"], 2 / 3)
        self.assertAlmostEqual(m["per_class"]["FALSE"]["precision"], 0.5)
        self.assertAlmostEqual(m["per_class"]["FALSE"]["f1"], 2 / 3)
        self.assertEqual(m["per_class"]["UNVERIFIABLE"]["f1"], 0.0)  # missed class counts as 0
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 2 / 3) / 3)  # all 3 classes averaged
        self.assertAlmostEqual(m["accuracy"], 0.6)

    def test_confusion_matrix_shape_and_orientation(self):
        cm = confusion_matrix(["TRUE", "FALSE"], ["FALSE", "FALSE"])
        self.assertEqual(len(cm), 3)
        self.assertTrue(all(len(r) == 3 for r in cm))
        self.assertEqual(cm[0][1], 1)  # truth TRUE, predicted FALSE
        self.assertEqual(cm[1][1], 1)
        self.assertEqual(sum(map(sum, cm)), 2)

    def test_perfect(self):
        y = LABELS * 4
        m = compute_metrics(y, y)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["accuracy"], 1.0)


def _fake_pipeline(predict):
    """Stands in for get_claim_verdicts: predict(claim_text) -> (label, search_ok)."""
    calls = []

    def fake(claims, use_cache):
        assert use_cache is False
        calls.append([c.claim_text for c in claims])
        out = {}
        for c in claims:
            label, ok = predict(c.claim_text)
            out[c.claim_id] = {
                "verdict": SimpleNamespace(verdict=label, confidence=0.5, summary="s"),
                "search_meta": {"sources_found": 2, "queries_succeeded": 3 if ok else 0},
                "search_error": None,
            }
        return out

    return fake, calls


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.data = os.path.join(self.tmp, "claims.csv")
        with open(self.data, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Claim_ID", "Claim", "Ground_Truth", "System_Prediction", "Confidence", "Correct", "Notes"])
            for i, lab in enumerate((LABELS * 4)[:10], start=1):
                w.writerow([f"C{i}", f"claim {i}", lab.lower(), "", "", "", "n"])
        self.out = os.path.join(self.tmp, "out")

    def _run(self, *extra):
        argv = ["run_eval", self.data, "--out", self.out, *extra]
        with mock.patch.object(sys, "argv", argv):
            run_eval.main()

    def test_end_to_end_resume_and_dataset_untouched(self):
        before = open(self.data, encoding="utf-8").read()
        truth = {f"claim {i}": lab for i, lab in enumerate((LABELS * 4)[:10], start=1)}
        fake, calls = _fake_pipeline(lambda text: (truth[text], True))

        with mock.patch("factcheck.pipeline.verdicts.get_claim_verdicts", fake):
            self._run()

        self.assertEqual(sum(len(c) for c in calls), 10)
        self.assertEqual(open(self.data, encoding="utf-8").read(), before)  # input unchanged
        import json
        m = json.load(open(os.path.join(self.out, "metrics.json")))
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["n"], 10)
        self.assertTrue(os.path.exists(os.path.join(self.out, "confusion_matrix.csv")))

        # second run reuses everything -> zero new pipeline calls
        fake2, calls2 = _fake_pipeline(lambda text: ("FALSE", True))
        with mock.patch("factcheck.pipeline.verdicts.get_claim_verdicts", fake2):
            self._run()
        self.assertEqual(calls2, [])

    def test_errored_rows_are_retried(self):
        fake, _ = _fake_pipeline(lambda text: ("UNVERIFIABLE", False))
        with mock.patch("factcheck.pipeline.verdicts.get_claim_verdicts", fake):
            self._run()
        rows = list(csv.DictReader(open(os.path.join(self.out, "predictions.csv"), encoding="utf-8")))
        self.assertTrue(all(r["Error"] == "search_failed" for r in rows))

        fake2, calls2 = _fake_pipeline(lambda text: ("TRUE", True))
        with mock.patch("factcheck.pipeline.verdicts.get_claim_verdicts", fake2):
            self._run()
        self.assertEqual(sum(len(c) for c in calls2), 10)

    def test_bad_label_rejected(self):
        with open(self.data, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["C99", "x", "MAYBE", "", "", "", ""])
        with self.assertRaises(SystemExit):
            run_eval.load_dataset(self.data)


if __name__ == "__main__":
    unittest.main()


class ThreeClassValidationTest(unittest.TestCase):
    def test_labels_exactly_three(self):
        self.assertEqual(LABELS, ["TRUE", "FALSE", "UNVERIFIABLE"])

    def test_retired_verdicts_rejected(self):
        from pydantic import ValidationError
        from factcheck.models import Verdict
        for old in ("PARTIALLY TRUE", "MISLEADING"):
            with self.assertRaises(ValidationError):
                Verdict(verdict=old, confidence=0.5, summary="s", explanation="e", cited_sources=[])

    def test_verdict_case_normalised(self):
        from factcheck.models import Verdict
        v = Verdict(verdict=" false ", confidence=0.5, summary="s", explanation="e", cited_sources=[])
        self.assertEqual(v.verdict, "FALSE")

    def test_dataset_validation_rejects_retired_ground_truth(self):
        import tempfile
        p = os.path.join(tempfile.mkdtemp(), "c.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["Claim_ID", "Claim", "Ground_Truth"])
            w.writerow(["1", "x", "PARTIALLY TRUE"])
        with self.assertRaises(SystemExit):
            run_eval.load_dataset(p)
