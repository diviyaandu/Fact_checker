"""
app.py — Phase 1 + Phase 2 + Phase 3 + Phase 4: Full pipeline through verdicts

UI/orchestration only — all business logic lives in pipeline.py.

Run with: streamlit run app.py
"""

import streamlit as st

from youtube import extract_video_id, is_valid_youtube_url
import pipeline
import cache_utils
from logging_utils import logger


st.set_page_config(page_title="Video Fact Checker (Prototype)", layout="centered")

st.title("🎥 YouTube Fact Checker — Prototype")
st.caption(
    "Full pipeline: transcript → claims → evidence search → "
    "evidence-grounded verdicts. "
    "This is an academic prototype, not a production system. "
    "Verdicts reflect the strength of retrieved evidence, not absolute truth."
)

with st.sidebar:
    st.subheader("⚙️ Cache")
    use_cache = st.checkbox(
        "Use cached results",
        value=True,
        help="Skips re-running the LLM/search pipeline for a video or claim "
        "you've already processed. Turn off for a clean baseline-vs-improved "
        "evaluation run.",
    )
    if st.button("Clear cache"):
        cache_utils.clear()
        st.success("Cache cleared.")

VERDICT_COLORS = {
    "TRUE": "🟢",
    "FALSE": "🔴",
    "PARTIALLY TRUE": "🟡",
    "MISLEADING": "🟠",
    "UNVERIFIABLE": "⚪",
}

url = st.text_input("Enter a YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

if st.button("Analyze Video", type="primary"):
    if not is_valid_youtube_url(url):
        st.error("That doesn't look like a valid YouTube URL. Try a watch?v= or /shorts/ link.")
    else:
        video_id = extract_video_id(url)
        st.write(f"**Video ID:** `{video_id}`")

        # STEP 1: Transcript (official captions, falling back to Whisper)
        cached_transcript = pipeline.get_cached_transcript(video_id, use_cache)

        if cached_transcript:
            segments = cached_transcript["segments"]
            method = cached_transcript["method"]
            st.caption("♻️ Using cached transcript for this video.")
        else:
            with st.spinner("Checking for official captions..."):
                segments = pipeline.fetch_official_transcript(video_id)

            method = "Official YouTube captions"

            if segments is None:
                st.warning("No official captions found. Falling back to local speech-to-text...")
                with st.spinner("Downloading audio and transcribing with Whisper..."):
                    try:
                        segments = pipeline.fetch_whisper_transcript(video_id)
                        method = "faster-whisper (local transcription)"
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
                        segments = None

            if segments:
                pipeline.save_transcript(video_id, segments, method)

        # STEP 2: Process transcript
        if segments:
            st.success(f"Transcript ready — method: {method}")

            with st.expander("📄 View full transcript"):
                for seg in segments:
                    st.markdown(f"**[{seg['start_str']} – {seg['end_str']}]**  {seg['text']}")

            # STEP 3: Extract claims
            st.subheader("🔍 Fact-Check Results")

            claims = pipeline.get_cached_claims(video_id, use_cache)

            if claims is None:
                with st.spinner("Identifying checkable claims..."):
                    claims = pipeline.fetch_claims(segments)
                pipeline.save_claims(video_id, claims)

            logger.info("CLAIMS ready | video_id=%s | count=%d", video_id, len(claims))

            if not claims:
                st.info("No checkable factual claims were identified in this video.")
            else:
                # STEP 4: Evidence search (all uncached claims) + batched
                # LLM verdicts (BATCH_SIZE claims per Groq call).
                with st.spinner(
                    f"Searching trusted sources and comparing {len(claims)} "
                    "claim(s) against evidence..."
                ):
                    claim_results = pipeline.get_claim_verdicts(claims, use_cache)

                verdict_counts = {}

                # STEP 5: Display each claim's result
                for claim in claims:
                    result = claim_results[claim.claim_id]
                    verdict = result["verdict"]

                    with st.container(border=True):
                        st.markdown(
                            f"**Claim #{claim.claim_id}** — `{claim.category}` · 🕐 "
                            f"{claim.start_time}–{claim.end_time}"
                        )
                        st.markdown(f"> {claim.claim_text}")

                        if result["from_cache"]:
                            st.caption("♻️ Using cached search + verdict for this claim.")

                        if result.get("search_error"):
                            st.error(f"Search failed: {result['search_error']}")

                        verdict_counts[verdict.verdict] = verdict_counts.get(verdict.verdict, 0) + 1
                        icon = VERDICT_COLORS.get(verdict.verdict, "⚪")

                        st.markdown(f"### {icon} {verdict.verdict}")
                        st.progress(
                            verdict.confidence,
                            text=f"Evidence confidence: {verdict.confidence:.0%}",
                        )
                        bd = verdict.confidence_breakdown or {}
                        if bd.get("components"):
                            with st.expander("How was this confidence calculated?"):
                                comps = bd["components"]
                                wts = bd["weights"]
                                st.caption(
                                    "Computed from evidence factors (0–1), not assigned by the model."
                                )
                                for name in ("relevance", "credibility", "strength", "agreement"):
                                    st.write(f"- {name.title()}: {comps[name]:.2f} (weight {wts[name]:.2f})")
                                if bd.get("conflict_capped"):
                                    st.write("- ⚠️ Conflicting evidence found — confidence capped.")
                        st.markdown(f"**{verdict.summary}**")
                        st.markdown(verdict.explanation)

                        if verdict.cited_sources:
                            st.markdown("**Evidence:**")
                            for src in verdict.cited_sources:
                                st.markdown(f"- **{src.publisher}** — [{src.title}]({src.url})")
                                st.caption(src.relevant_text[:200] + "...")
                        else:
                            st.caption("No verifiably relevant evidence was cited for this claim.")

                # STEP 6: Summary
                st.divider()
                st.subheader("📊 Summary")
                st.write(f"**{len(claims)} claims analyzed:**")

                summary_line = " · ".join(
                    f"{VERDICT_COLORS.get(v, '')} {v}: {c}" for v, c in verdict_counts.items()
                )
                st.write(summary_line)
                logger.info(
                    "PIPELINE complete | video_id=%s | claims=%d | verdicts=%s",
                    video_id, len(claims), verdict_counts,
                )

                st.caption(
                    "This summary reflects the distribution of verdicts based on retrieved "
                    "evidence — it is not a scientific 'truth score' for the video."
                )
        else:
            st.error("Could not produce a transcript for this video.")