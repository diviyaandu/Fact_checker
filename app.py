"""
app.py — Phase 1 + Phase 2 + Phase 3 + Phase 4: Full pipeline through verdicts

Run with: streamlit run app.py
"""

import streamlit as st

from youtube import extract_video_id, is_valid_youtube_url, get_official_transcript
from transcription import transcribe_with_whisper
from claims import extract_claims
from search import search_evidence_for_claim
from fact_checker import fact_check_claim


st.set_page_config(page_title="Video Fact Checker (Prototype)", layout="centered")

st.title("🎥 YouTube Fact Checker — Prototype")
st.caption(
    "Full pipeline: transcript → claims → evidence search → "
    "evidence-grounded verdicts. "
    "This is an academic prototype, not a production system. "
    "Verdicts reflect the strength of retrieved evidence, not absolute truth."
)

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

        # STEP 1: Get official YouTube transcript
        with st.spinner("Checking for official captions..."):
            segments = get_official_transcript(video_id)

        method = "Official YouTube captions"

        # STEP 2: Fallback to Whisper if captions are unavailable
        if segments is None:
            st.warning("No official captions found. Falling back to local speech-to-text...")
            with st.spinner("Downloading audio and transcribing with Whisper..."):
                try:
                    segments = transcribe_with_whisper(video_id)
                    method = "faster-whisper (local transcription)"
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
                    segments = None

        # STEP 3: Process transcript
        if segments:
            st.success(f"Transcript ready — method: {method}")

            with st.expander("📄 View full transcript"):
                for seg in segments:
                    st.markdown(f"**[{seg['start_str']} – {seg['end_str']}]**  {seg['text']}")

            # STEP 4: Extract claims
            st.subheader("🔍 Fact-Check Results")

            with st.spinner("Identifying checkable claims..."):
                claims = extract_claims(segments)

            if not claims:
                st.info("No checkable factual claims were identified in this video.")
            else:
                verdict_counts = {}

                # STEP 5: Analyze each claim
                for claim in claims:
                    with st.container(border=True):
                        st.markdown(
                            f"**Claim #{claim.claim_id}** — `{claim.category}` · 🕐 "
                            f"{claim.start_time}–{claim.end_time}"
                        )
                        st.markdown(f"> {claim.claim_text}")

                                                # STEP 5A: Search evidence
                        with st.spinner("Searching trusted sources..."):
                            try:
                                found_sources, search_meta = search_evidence_for_claim(
                                    claim.claim_text, claim.category
                                )
                            except Exception as e:
                                st.error(f"Search failed: {e}")
                                found_sources, search_meta = [], {}

                        # STEP 5B: Fact-check claim
                        with st.spinner("Comparing claim against evidence..."):
                            verdict = fact_check_claim(claim.claim_text, found_sources, search_meta)

                        # STEP 5C: Count verdicts
                        verdict_counts[verdict.verdict] = verdict_counts.get(verdict.verdict, 0) + 1
                        icon = VERDICT_COLORS.get(verdict.verdict, "⚪")

                        # STEP 5D: Display verdict
                        st.markdown(f"### {icon} {verdict.verdict}")
                        st.progress(
                            verdict.confidence,
                            text=f"Evidence confidence: {verdict.confidence:.0%}",
                        )
                        st.markdown(f"**{verdict.summary}**")
                        st.markdown(verdict.explanation)

                        # STEP 5E: Display cited evidence
                                                # STEP 5E: Display cited evidence
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

                st.caption(
                    "This summary reflects the distribution of verdicts based on retrieved "
                    "evidence — it is not a scientific 'truth score' for the video."
                )
        else:
            st.error("Could not produce a transcript for this video.")