"""All st.session_state access for the front end, in one place."""

import streamlit as st

from ui import config
from ui.timing import cooldown_remaining, now

# Key names are part of the tests' contract; change them in both places.
_COOLDOWN_STARTED = "analyze_cooldown_started"   # time.monotonic() of the click
_PENDING_URL = "analyze_pending_url"             # URL waiting to be analysed
_LAST_ANALYSIS = "last_analysis"                 # finished analysis, re-drawn on every rerun


# ---- cooldown -------------------------------------------------------------

def cooldown_seconds_left() -> float:
    return cooldown_remaining(
        st.session_state.get(_COOLDOWN_STARTED), config.ANALYZE_COOLDOWN_SECONDS
    )


def start_cooldown() -> None:
    st.session_state[_COOLDOWN_STARTED] = now()


def clear_cooldown_if_expired() -> None:
    if cooldown_seconds_left() <= 0:
        st.session_state.pop(_COOLDOWN_STARTED, None)


# ---- queued analysis ------------------------------------------------------

def queue_analysis(url: str) -> None:
    """Remember the URL to analyse on the next run and drop the previous results."""
    st.session_state[_PENDING_URL] = url
    st.session_state.pop(_LAST_ANALYSIS, None)


def pop_queued_analysis() -> str | None:
    return st.session_state.pop(_PENDING_URL, None)


# ---- finished analysis ----------------------------------------------------

def save_analysis(analysis: dict) -> None:
    st.session_state[_LAST_ANALYSIS] = analysis


def last_analysis() -> dict | None:
    return st.session_state.get(_LAST_ANALYSIS)
