"""Pure cooldown arithmetic (no Streamlit, easy to unit-test)."""

import time


def now() -> float:
    """The clock used for cooldowns (monotonic, so wall-clock changes can't affect it)."""
    return time.monotonic()


def cooldown_remaining(started_at: float | None, total: float, now: float | None = None) -> float:
    """Seconds left of the cooldown (0.0 when it never started or has ended).
    `started_at` must come from the same clock as `now` (default time.monotonic)."""
    if started_at is None or total <= 0:
        return 0.0
    now = time.monotonic() if now is None else now
    return max(0.0, total - (now - started_at))
