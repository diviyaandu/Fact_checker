"""UI settings. Everything tunable about the front end lives here."""

import os


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# Cooldown after an Analyze click that needs fresh Groq/Tavily calls. Fully cached
# analyses are exempt. Override with e.g.
#   ANALYZE_COOLDOWN_SECONDS=30 streamlit run app.py      (0 disables it)
ANALYZE_COOLDOWN_SECONDS = _env_float("ANALYZE_COOLDOWN_SECONDS", 60.0)

# How long the "Cache cleared" notice stays before it fades away.
CACHE_NOTICE_SECONDS = 3.5

# Stable widget key -> Streamlit adds the CSS class .st-key-analyze_btn to the
# button, which is how the cooldown fill is attached to it (needs streamlit>=1.39).
ANALYZE_BUTTON_KEY = "analyze_btn"
