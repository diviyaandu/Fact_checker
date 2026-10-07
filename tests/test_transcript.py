"""Transcript helpers: URL parsing, segment format, official captions (API mocked)."""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _stubs  # noqa: E402

_stubs.install()

from factcheck.transcript import extract_video_id, is_valid_youtube_url  # noqa: E402
from factcheck.transcript import youtube  # noqa: E402
from factcheck.transcript.segments import make_segment, seconds_to_mmss  # noqa: E402

VID = "dQw4w9WgXcQ"


class UrlParsing(unittest.TestCase):
    def test_supported_url_shapes(self):
        for url in (f"https://www.youtube.com/watch?v={VID}", f"https://youtu.be/{VID}",
                    f"https://www.youtube.com/shorts/{VID}", f"youtube.com/watch?v={VID}&t=5s"):
            self.assertEqual(extract_video_id(url), VID, url)
            self.assertTrue(is_valid_youtube_url(url))

    def test_rejects_everything_else(self):
        for url in ("", "not a url", "https://example.com/watch?v=" + VID,
                    "https://www.youtube.com/watch?v=short"):
            self.assertIsNone(extract_video_id(url), url)
            self.assertFalse(is_valid_youtube_url(url))


class Segments(unittest.TestCase):
    def test_mmss(self):
        self.assertEqual(seconds_to_mmss(0), "00:00")
        self.assertEqual(seconds_to_mmss(65.9), "01:05")
        self.assertEqual(seconds_to_mmss(3600), "60:00")

    def test_make_segment_shape(self):
        self.assertEqual(make_segment(1.5, 70, "  hi  "), {
            "start": 1.5, "end": 70, "start_str": "00:01", "end_str": "01:10", "text": "hi"})


class OfficialCaptions(unittest.TestCase):
    def test_captions_become_segments(self):
        raw = [{"start": 0.0, "duration": 2.5, "text": " hello "}, {"start": 3.0, "text": "no duration"}]
        with mock.patch.object(youtube, "YouTubeTranscriptApi") as api:
            api.get_transcript.return_value = raw
            segs = youtube.get_official_transcript(VID)
        self.assertEqual(segs[0], make_segment(0.0, 2.5, "hello"))
        self.assertEqual(segs[1]["end"], 3.0)           # missing duration counts as 0

    def test_any_failure_means_none(self):
        class Disabled(Exception):
            pass

        # Real exception classes, so this also holds when the stand-in SDK is used.
        with mock.patch.object(youtube, "YouTubeTranscriptApi") as api, \
                mock.patch.object(youtube, "TranscriptsDisabled", Disabled), \
                mock.patch.object(youtube, "NoTranscriptFound", Disabled), \
                mock.patch.object(youtube, "VideoUnavailable", Disabled):
            for error in (Disabled("no captions"), RuntimeError("blocked")):
                api.get_transcript.side_effect = error
                self.assertIsNone(youtube.get_official_transcript(VID))


if __name__ == "__main__":
    unittest.main()
