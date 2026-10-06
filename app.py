"""
app.py — Streamlit entry point for the YouTube Fact Checker.

UI/orchestration only; the layout is split across the ui/ package and all
business logic lives in pipeline.py.

Run with: streamlit run app.py   (needs streamlit >= 1.39)
"""

import streamlit as st

from ui.sidebar import render_sidebar
from ui.controls import render_analyze_controls
from ui.results import render_analysis
from ui import state

st.set_page_config(page_title="Video Fact Checker (Prototype)", layout="centered")

st.title("🎥 YouTube Fact Checker — Prototype")
st.caption(
    "Full pipeline: transcript → claims → evidence search → "
    "evidence-grounded verdicts. "
    "This is an academic prototype, not a production system. "
    "Verdicts reflect the strength of retrieved evidence, not absolute truth."
)

use_cache = render_sidebar()
render_analyze_controls(use_cache)

analysis = state.last_analysis()
if analysis:
    render_analysis(analysis)
