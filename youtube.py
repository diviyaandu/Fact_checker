"""
youtube.py

Handles:
- Validating YouTube URLs
- Extracting the video ID
- Fetching official captions via youtube-transcript-api (preferred, no download needed)
"""

import re
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

YOUTUBE_URL_PATTERNS = [
    r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
    r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
    r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
]


def extract_video_id(url: str) -> str | None:
    """Return the 11-character YouTube video ID, or None if the URL is not valid."""
    for pattern in YOUTUBE_URL_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def is_valid_youtube_url(url: str) -> bool:
    return extract_video_id(url) is not None


def _seconds_to_mmss(seconds: float) -> str:
    total = int(seconds)
    m, s = divmod(total, 60)
    return f"{m:02d}:{s:02d}"


def get_official_transcript(video_id: str) -> list[dict] | None:
    """
    Try to fetch YouTube's own captions.
    Returns a list of {start, end, text} dicts, or None if unavailable.
    This does NOT download the video — it only reads publicly served caption data.
    """
    try:
        raw = YouTubeTranscriptApi.get_transcript(video_id)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        return None
    except Exception:
        return None

    segments = []
    for entry in raw:
        start = entry["start"]
        end = start + entry.get("duration", 0)
        segments.append(
            {
                "start": start,
                "end": end,
                "start_str": _seconds_to_mmss(start),
                "end_str": _seconds_to_mmss(end),
                "text": entry["text"].strip(),
            }
        )
    return segments