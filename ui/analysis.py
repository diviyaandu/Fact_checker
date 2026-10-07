"""
Runs the existing pipeline for one URL and returns everything the results view
needs. Only spinners are drawn here; the results themselves are drawn by
ui.results so they can be re-drawn on any rerun from session state.
"""

import streamlit as st

from factcheck import pipeline
from factcheck.logging_utils import logger
from factcheck.transcript import extract_video_id


def verdict_counts(claims, claim_results) -> dict[str, int]:
    counts: dict[str, int] = {}
    for claim in claims:
        v = claim_results[claim.claim_id]["verdict"].verdict
        counts[v] = counts.get(v, 0) + 1
    return counts


def run_analysis(url: str, use_cache: bool) -> dict:
    video_id = extract_video_id(url)
    out = {
        "video_id": video_id,
        "transcript_cached": False,
        "fell_back_to_whisper": False,
        "transcription_error": None,
        "segments": None,
        "method": None,
        "claims": None,
        "claim_results": None,
    }

    segments, method = _get_transcript(video_id, use_cache, out)
    out["segments"], out["method"] = segments, method
    if not segments:
        return out

    claims = _get_claims(video_id, segments, use_cache)
    out["claims"] = claims

    if claims:
        # Evidence search (all uncached claims) + batched LLM verdicts
        # (BATCH_SIZE claims per Groq call).
        with st.spinner(
            f"Searching trusted sources and comparing {len(claims)} "
            "claim(s) against evidence..."
        ):
            out["claim_results"] = pipeline.get_claim_verdicts(claims, use_cache)

        logger.info(
            "PIPELINE complete | video_id=%s | claims=%d | verdicts=%s",
            video_id, len(claims), verdict_counts(claims, out["claim_results"]),
        )

    return out


def _get_transcript(video_id: str, use_cache: bool, out: dict):
    """Official captions, falling back to Whisper. Notes what happened in `out`."""
    cached = pipeline.get_cached_transcript(video_id, use_cache)
    if cached:
        out["transcript_cached"] = True
        return cached["segments"], cached["method"]

    with st.spinner("Checking for official captions..."):
        segments = pipeline.fetch_official_transcript(video_id)

    method = "Official YouTube captions"

    if segments is None:
        out["fell_back_to_whisper"] = True
        with st.spinner(
            "No official captions found. Downloading audio and transcribing with Whisper..."
        ):
            try:
                segments = pipeline.fetch_whisper_transcript(video_id)
                method = "faster-whisper (local transcription)"
            except Exception as e:
                out["transcription_error"] = str(e)
                segments = None

    if segments:
        pipeline.save_transcript(video_id, segments, method)

    return segments, method


def _get_claims(video_id: str, segments: list[dict], use_cache: bool):
    claims = pipeline.get_cached_claims(video_id, use_cache)

    if claims is None:
        with st.spinner("Identifying checkable claims..."):
            claims = pipeline.fetch_claims(segments)
        pipeline.save_claims(video_id, claims)

    logger.info("CLAIMS ready | video_id=%s | count=%d", video_id, len(claims))
    return claims
