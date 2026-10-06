"""Draws a finished analysis (see ui.analysis.run_analysis for its shape)."""

import streamlit as st

from ui.analysis import verdict_counts

VERDICT_COLORS = {
    "TRUE": "🟢",
    "FALSE": "🔴",
    "PARTIALLY TRUE": "🟡",
    "MISLEADING": "🟠",
    "UNVERIFIABLE": "⚪",
}


def render_analysis(a: dict) -> None:
    st.write(f"**Video ID:** `{a['video_id']}`")

    if a["transcript_cached"]:
        st.caption("♻️ Using cached transcript for this video.")
    if a["fell_back_to_whisper"]:
        st.warning("No official captions found. Falling back to local speech-to-text...")
    if a["transcription_error"]:
        st.error(f"Transcription failed: {a['transcription_error']}")

    if not a["segments"]:
        st.error("Could not produce a transcript for this video.")
        return

    _render_transcript(a["segments"], a["method"])

    st.subheader("🔍 Fact-Check Results")

    claims = a["claims"]
    if not claims:
        st.info("No checkable factual claims were identified in this video.")
        return

    for claim in claims:
        _render_claim(claim, a["claim_results"][claim.claim_id])

    _render_summary(claims, a["claim_results"])


def _render_transcript(segments: list[dict], method: str) -> None:
    st.success(f"Transcript ready — method: {method}")

    with st.expander("📄 View full transcript"):
        for seg in segments:
            st.markdown(f"**[{seg['start_str']} – {seg['end_str']}]**  {seg['text']}")


def _render_claim(claim, result: dict) -> None:
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

        icon = VERDICT_COLORS.get(verdict.verdict, "⚪")

        st.markdown(f"### {icon} {verdict.verdict}")
        st.progress(
            verdict.confidence,
            text=f"Evidence confidence: {verdict.confidence:.0%}",
        )
        _render_confidence_breakdown(verdict.confidence_breakdown or {})
        st.markdown(f"**{verdict.summary}**")
        st.markdown(verdict.explanation)

        _render_evidence(verdict.cited_sources)


def _render_confidence_breakdown(bd: dict) -> None:
    if not bd.get("components"):
        return

    with st.expander("How was this confidence calculated?"):
        comps, wts = bd["components"], bd["weights"]
        st.caption("Computed from evidence factors (0–1), not assigned by the model.")
        for name in ("relevance", "credibility", "strength", "agreement"):
            st.write(f"- {name.title()}: {comps[name]:.2f} (weight {wts[name]:.2f})")
        if bd.get("conflict_capped"):
            st.write("- ⚠️ Conflicting evidence found — confidence capped.")


def _render_evidence(cited_sources) -> None:
    if not cited_sources:
        st.caption("No verifiably relevant evidence was cited for this claim.")
        return

    st.markdown("**Evidence:**")
    for src in cited_sources:
        st.markdown(f"- **{src.publisher}** — [{src.title}]({src.url})")
        st.caption(src.relevant_text[:200] + "...")


def _render_summary(claims, claim_results) -> None:
    st.divider()
    st.subheader("📊 Summary")
    st.write(f"**{len(claims)} claims analyzed:**")

    summary_line = " · ".join(
        f"{VERDICT_COLORS.get(v, '')} {v}: {c}"
        for v, c in verdict_counts(claims, claim_results).items()
    )
    st.write(summary_line)

    st.caption(
        "This summary reflects the distribution of verdicts based on retrieved "
        "evidence — it is not a scientific 'truth score' for the video."
    )
