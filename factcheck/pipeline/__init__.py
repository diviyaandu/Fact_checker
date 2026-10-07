"""
All non-UI business logic for one analysis, stage by stage:

    transcripts.py   transcript: cache -> official captions -> Whisper fallback
    claims.py        claim extraction (max MAX_CLAIMS) with caching
    verdicts.py      evidence search + batched LLM verdicts + caching
    cache_check.py   "is this whole video already cached?" (no API calls)

The UI should only use the names exported here.
"""

from factcheck.pipeline.cache_check import is_fully_cached
from factcheck.pipeline.claims import fetch_claims, get_cached_claims, save_claims
from factcheck.pipeline.transcripts import (
    fetch_official_transcript,
    fetch_whisper_transcript,
    get_cached_transcript,
    save_transcript,
)
from factcheck.pipeline.verdicts import get_claim_verdicts

__all__ = [
    "is_fully_cached",
    "fetch_claims",
    "get_cached_claims",
    "save_claims",
    "fetch_official_transcript",
    "fetch_whisper_transcript",
    "get_cached_transcript",
    "save_transcript",
    "get_claim_verdicts",
]
