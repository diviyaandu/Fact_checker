"""
Tests for the front end: cooldown, cached-analysis bypass, fading cache notice.

Streamlit is replaced by a scripted fake so app.py (and the ui/ package) can be
executed top to bottom once per simulated rerun — no browser, no network:
    python -m unittest discover -s tests -v
"""

import os
import sys
import time
import types
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.timing import cooldown_remaining  # noqa: E402  (pure modules, no Streamlit)
from ui.styles import cooldown_button_css, cache_notice_html  # noqa: E402

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
BUTTON = "Analyze Video"


class CooldownMath(unittest.TestCase):
    def test_remaining(self):
        self.assertEqual(cooldown_remaining(None, 60), 0.0)
        self.assertEqual(cooldown_remaining(100.0, 60, now=100.0), 60.0)
        self.assertEqual(cooldown_remaining(100.0, 60, now=130.0), 30.0)
        self.assertEqual(cooldown_remaining(100.0, 60, now=160.0), 0.0)
        self.assertEqual(cooldown_remaining(100.0, 60, now=500.0), 0.0)
        self.assertEqual(cooldown_remaining(100.0, 0, now=101.0), 0.0)   # 0 disables it

    def test_css_resumes_from_current_progress(self):
        css = cooldown_button_css(remaining=15, total=60, key="analyze_btn", primary="#ff4b4b")
        self.assertIn(".st-key-analyze_btn button", css)
        self.assertIn("scaleX(0.25000)", css)          # 75% already filled
        self.assertIn("15.000s linear forwards", css)  # ...finishing in the remaining 15s
        self.assertIn("transform-origin: right center", css)  # reveals left -> right
        self.assertIn("background-color: #ff4b4b !important", css)
        self.assertNotRegex(css, r"counter|attr\(")    # no numeric countdown

    def test_css_rejects_unsafe_colour_and_clamps(self):
        css = cooldown_button_css(999, 60, "k", primary="red; } body { display:none")
        self.assertNotIn("display:none", css)
        self.assertIn("#FF4B4B", css)
        self.assertIn("scaleX(1.00000)", css)

    def test_notice_fades_and_restarts_each_time(self):
        a = cache_notice_html("Cache cleared.", 3.5)
        self.assertIn("Cache cleared.", a)
        self.assertIn("visibility: hidden", a)
        self.assertIn("3.50s", a)
        with mock.patch("ui.styles.time.time", return_value=time.time() + 5):
            b = cache_notice_html("Cache cleared.", 3.5)
        self.assertNotEqual(a, b)   # unique animation name -> replays on a second click


class Rerun(Exception):
    pass


class FakeStreamlit(types.ModuleType):
    """Just enough of Streamlit for app.py. `clicks` = labels clicked this run."""

    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.clicks = set()
        self.url = "https://www.youtube.com/watch?v=AAAAAAAAAAA"
        self.button_calls = {}
        self.markdown_calls = []
        self.fragments = []
        self.progress_calls = 0
        self.errors = []
        self._m = mock.MagicMock()

    def button(self, label, **kw):
        self.button_calls[label] = kw
        return label in self.clicks

    def text_input(self, *a, **k):
        return self.url

    def checkbox(self, *a, **k):
        return True

    def markdown(self, body, *a, **k):
        self.markdown_calls.append(body)

    def progress(self, *a, **k):
        self.progress_calls += 1

    def error(self, msg, *a, **k):
        self.errors.append(msg)

    def rerun(self, *a, **k):
        raise Rerun()

    def fragment(self, **kw):
        def deco(fn):
            def call():
                self.fragments.append((kw, fn))
            return call
        return deco

    def get_option(self, name):
        return None

    def __getattr__(self, name):          # sidebar, spinner, container, expander, caption, ...
        return getattr(self._m, name)


def make_claim_results():
    verdict = SimpleNamespace(
        verdict="TRUE", confidence=0.8, summary="s", explanation="e",
        cited_sources=[], confidence_breakdown=None,
    )
    claim = SimpleNamespace(claim_id=1, claim_text="c", category="Science",
                            start_time="0:00", end_time="0:10")
    return [claim], {1: {"verdict": verdict, "from_cache": False, "search_error": None}}


class AppFlow(unittest.TestCase):
    def setUp(self):
        self.st = FakeStreamlit()
        self.cached = False        # what pipeline.is_fully_cached reports
        claims, results = make_claim_results()
        self.pipeline = mock.MagicMock()
        self.pipeline.is_fully_cached.side_effect = lambda video_id, use_cache: self.cached
        self.pipeline.get_cached_transcript.return_value = None
        self.pipeline.fetch_official_transcript.return_value = [
            {"start_str": "0:00", "end_str": "0:05", "text": "hello"}]
        self.pipeline.get_cached_claims.return_value = None
        self.pipeline.fetch_claims.return_value = claims
        self.pipeline.get_claim_verdicts.return_value = results

        youtube = types.ModuleType("youtube")
        youtube.extract_video_id = lambda u: "AAAAAAAAAAA"
        youtube.is_valid_youtube_url = lambda u: "youtube.com" in u

        self.patcher = mock.patch.dict(sys.modules, {
            "streamlit": self.st, "pipeline": self.pipeline, "youtube": youtube,
            "cache_utils": mock.MagicMock(),
        })
        self.patcher.start()
        # Fresh ui package per test, so it binds to THIS test's fake Streamlit
        # (patch.dict restores the original sys.modules afterwards).
        for name in [m for m in sys.modules if m == "ui" or m.startswith("ui.")]:
            del sys.modules[name]
        with open(APP_PATH, encoding="utf-8") as f:
            self.source = f.read()

    def tearDown(self):
        self.patcher.stop()

    def run_app(self, *clicks):
        self.st.clicks = set(clicks)
        self.st.markdown_calls.clear()
        self.st.fragments.clear()
        self.st.progress_calls = 0
        exec(compile(self.source, APP_PATH, "exec"), {"__name__": "__main__"})

    def expire_cooldown(self):
        self.st.session_state["analyze_cooldown_started"] -= 1000

    def fill_css_shown(self):
        return any("@keyframes cdfill" in m for m in self.st.markdown_calls)

    def disabled(self):
        return self.st.button_calls[BUTTON]["disabled"]

    # ------------------------------------------------------------------
    def test_full_cooldown_lifecycle(self):
        # Fresh page: enabled, no cooldown styling, no watcher.
        self.run_app()
        self.assertFalse(self.disabled())
        self.assertEqual(self.st.fragments, [])

        # Click: starts cooldown, stores URL, reruns straight away (no analysis yet).
        with self.assertRaises(Rerun):
            self.run_app(BUTTON)
        self.assertIn("analyze_cooldown_started", self.st.session_state)
        self.assertEqual(self.pipeline.get_claim_verdicts.call_count, 0)

        # The rerun: button disabled + fill CSS + re-enable watcher; analysis runs once.
        self.run_app()
        self.assertTrue(self.disabled())
        self.assertTrue(self.fill_css_shown())
        self.assertEqual(len(self.st.fragments), 1)
        self.assertEqual(self.pipeline.get_claim_verdicts.call_count, 1)
        self.assertEqual(self.st.progress_calls, 1)

        # A later ordinary rerun mid-cooldown: still disabled, results still shown,
        # pipeline NOT re-run.
        self.run_app()
        self.assertTrue(self.disabled())
        self.assertEqual(self.pipeline.get_claim_verdicts.call_count, 1)
        self.assertEqual(self.st.progress_calls, 1)

        # Clicking while cooling is ignored.
        self.run_app(BUTTON)
        self.assertEqual(self.pipeline.get_claim_verdicts.call_count, 1)

        # Watcher does nothing mid-cooldown, but reruns the app once it has ended.
        _kw, watcher = self.st.fragments[0]
        watcher()
        self.expire_cooldown()
        with self.assertRaises(Rerun):
            watcher()

        # After expiry: enabled again, no cooldown CSS, results from before still there.
        self.run_app()
        self.assertFalse(self.disabled())
        self.assertFalse(self.fill_css_shown())
        self.assertEqual(self.st.fragments, [])
        self.assertEqual(self.st.progress_calls, 1)
        self.assertNotIn("analyze_cooldown_started", self.st.session_state)

    def test_invalid_url_does_not_start_cooldown(self):
        self.st.url = "not a link"
        self.run_app(BUTTON)
        self.assertNotIn("analyze_cooldown_started", self.st.session_state)
        self.assertTrue(self.st.errors)

    def test_cache_notice_only_in_click_run(self):
        self.run_app("Clear cache")
        self.assertTrue(any("Cache cleared." in m for m in self.st.markdown_calls))
        self.run_app()   # any later rerun: nothing left behind
        self.assertFalse(any("Cache cleared." in m for m in self.st.markdown_calls))

    # ---- cached analyses are exempt from the cooldown -------------------
    def test_fully_cached_analysis_never_starts_cooldown(self):
        self.cached = True
        with self.assertRaises(Rerun):
            self.run_app(BUTTON)
        self.assertNotIn("analyze_cooldown_started", self.st.session_state)

        self.run_app()   # the queued analysis runs, button stays usable
        self.assertFalse(self.disabled())
        self.assertFalse(self.fill_css_shown())
        self.assertEqual(self.st.fragments, [])
        self.assertEqual(self.pipeline.get_claim_verdicts.call_count, 1)
        self.assertEqual(self.st.progress_calls, 1)

        # ...and it can be clicked again immediately.
        with self.assertRaises(Rerun):
            self.run_app(BUTTON)

    def test_cached_url_is_allowed_during_cooldown_without_resetting_it(self):
        with self.assertRaises(Rerun):          # uncached run -> cooldown starts
            self.run_app(BUTTON)
        self.run_app()
        started = self.st.session_state["analyze_cooldown_started"]
        self.assertTrue(self.disabled())

        self.cached = True                      # user switches to a fully cached video
        self.run_app()
        self.assertFalse(self.disabled())
        self.assertFalse(self.fill_css_shown())

        with self.assertRaises(Rerun):          # click is honoured...
            self.run_app(BUTTON)
        self.run_app()
        self.assertEqual(self.pipeline.get_claim_verdicts.call_count, 2)
        self.assertEqual(self.st.session_state["analyze_cooldown_started"], started)  # ...cooldown untouched

        self.cached = False                     # back to an uncached URL: locked and filling again
        self.run_app()
        self.assertTrue(self.disabled())
        self.assertTrue(self.fill_css_shown())

    def test_cache_check_receives_use_cache_setting(self):
        self.run_app()
        self.pipeline.is_fully_cached.assert_called_with("AAAAAAAAAAA", True)


if __name__ == "__main__":
    unittest.main()
