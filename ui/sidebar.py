"""Sidebar: cache controls."""

import streamlit as st

import cache_utils
from ui import config
from ui.styles import cache_notice_html


def render_sidebar() -> bool:
    """Draws the cache controls and returns the 'use cached results' setting."""
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
            # Rendered only in the run triggered by this click, so it can't go
            # stale on later reruns; CSS fades it out and collapses it.
            st.markdown(
                cache_notice_html("Cache cleared.", config.CACHE_NOTICE_SECONDS),
                unsafe_allow_html=True,
            )

    return use_cache
