"""
URL box + Analyze button + cooldown.

Flow
----
 click (valid URL)  ->  queue the URL, start the cooldown *unless the whole
                        analysis is already cached*, rerun immediately
 next run           ->  button is drawn disabled and filling, then the queued
                        analysis runs and its result is stored in session state
 any later run      ->  result is re-drawn from session state (never re-run)
 cooldown ends      ->  a tiny fragment triggers one rerun; button is enabled again

Cached analyses cost no API calls, so they are exempt: the cooldown is never
started for them, and while one is running for *other* URLs the button stays
enabled whenever the URL in the box is fully cached.
"""

import streamlit as st

import pipeline
from youtube import extract_video_id, is_valid_youtube_url
from ui import config, state
from ui.analysis import run_analysis
from ui.styles import cooldown_button_css


def _url_is_fully_cached(url: str, use_cache: bool) -> bool:
    return bool(
        url
        and is_valid_youtube_url(url)
        and pipeline.is_fully_cached(extract_video_id(url), use_cache)
    )


def _reenable_when_done() -> None:
    # Runs inside a fragment: when the cooldown has ended, rerun the whole app once.
    if state.cooldown_seconds_left() <= 0:
        st.rerun()


def _show_cooldown_fill(remaining: float) -> None:
    # Injected into the sidebar so the style-only element leaves no gap in the
    # main column.
    css = cooldown_button_css(
        remaining,
        config.ANALYZE_COOLDOWN_SECONDS,
        config.ANALYZE_BUTTON_KEY,
        st.get_option("theme.primaryColor"),
    )
    with st.sidebar:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_analyze_controls(use_cache: bool) -> None:
    url = st.text_input("Enter a YouTube URL", placeholder="https://www.youtube.com/watch?v=...")

    state.clear_cooldown_if_expired()
    remaining = state.cooldown_seconds_left()
    cooling = remaining > 0

    cached = _url_is_fully_cached(url, use_cache)
    locked = cooling and not cached          # cached analyses ignore the cooldown

    clicked = st.button(
        "Analyze Video", type="primary", key=config.ANALYZE_BUTTON_KEY, disabled=locked
    )

    if locked:
        _show_cooldown_fill(remaining)

    if cooling:
        # One full rerun when the cooldown ends so the button re-enables.
        st.fragment(run_every=remaining + 0.25)(_reenable_when_done)()

    if clicked and not locked:
        if not is_valid_youtube_url(url):
            st.error("That doesn't look like a valid YouTube URL. Try a watch?v= or /shorts/ link.")
        else:
            if not cached:
                state.start_cooldown()       # only runs that hit Groq/Tavily/YouTube
            state.queue_analysis(url)
            st.rerun()

    queued_url = state.pop_queued_analysis()
    if queued_url:
        state.save_analysis(run_analysis(queued_url, use_cache))
