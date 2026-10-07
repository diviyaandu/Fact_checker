"""Getting a transcript for a YouTube video (official captions, Whisper fallback)."""

from factcheck.transcript.youtube import (
    extract_video_id,
    is_valid_youtube_url,
    get_official_transcript,
)
from factcheck.transcript.whisper import transcribe_with_whisper

__all__ = [
    "extract_video_id",
    "is_valid_youtube_url",
    "get_official_transcript",
    "transcribe_with_whisper",
]
