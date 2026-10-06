"""
Pure CSS / HTML builders for the front end (no Streamlit import).

  * cooldown_button_css()  - CSS that "fills" the disabled Analyze button left -> right
  * cache_notice_html()    - a message that fades out and collapses by itself
"""

import re
import time

DEFAULT_PRIMARY = "#FF4B4B"  # Streamlit's default primary colour
_HEX_COLOUR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def _safe_colour(colour: str | None) -> str:
    # The value is interpolated into CSS, so only accept plain hex colours.
    return colour if colour and _HEX_COLOUR.match(colour) else DEFAULT_PRIMARY


def cooldown_button_css(remaining: float, total: float, key: str, primary: str | None = None) -> str:
    """
    CSS (without <style> tags) for a st.button(key=...) that is cooling down.

    The disabled button keeps its normal (primary) colour. A dark overlay covers
    the part that is not "ready" yet and shrinks from full width to zero, anchored
    to the RIGHT edge, so the normal colour is revealed LEFT -> RIGHT. When the
    overlay reaches zero width the button is 100% filled (and the app re-enables it).

    The animation is defined from the *current* progress to 100% over the time
    remaining, so a Streamlit rerun mid-cooldown resumes at the right position.
    The keyframes name is unique per call so the browser restarts the animation
    with the new values instead of reusing the old one.
    """
    total = max(total, 0.001)
    remaining = min(max(remaining, 0.0), total)
    start_scale = remaining / total  # 1.0 = nothing filled yet, 0.0 = fully filled
    colour = _safe_colour(primary)
    anim = f"cdfill{int(time.time() * 1000)}"
    sel = f".st-key-{key} button"

    return f"""
@keyframes {anim} {{
  from {{ transform: scaleX({start_scale:.5f}); }}
  to   {{ transform: scaleX(0); }}
}}
{sel} {{
  position: relative;
  overflow: hidden;
}}
{sel}:disabled {{
  background-color: {colour} !important;
  border-color: {colour} !important;
  color: #fff !important;
  opacity: 1 !important;
  cursor: not-allowed !important;
}}
{sel} > div {{
  position: relative;
  z-index: 2;
}}
{sel}::after {{
  content: "";
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.55);
  transform-origin: right center;
  transform: scaleX({start_scale:.5f});
  animation: {anim} {remaining:.3f}s linear forwards;
  pointer-events: none;
  z-index: 1;
}}
"""


def cache_notice_html(text: str, seconds: float = 3.5) -> str:
    """
    A success-style notice that fades out and collapses after `seconds` using
    CSS only (no sleeping / extra reruns). The caller renders it only in the run
    triggered by the click, so it can never linger across later reruns; even if
    the run is still showing it, it is invisible and takes no space once faded.

    The animation name carries a timestamp so a second click always replays it.
    """
    name = f"cachefade{int(time.time() * 1000)}"
    return f"""
<style>
@keyframes {name} {{
  0%   {{ opacity: 0; transform: translateY(-4px); max-height: 4rem; }}
  8%   {{ opacity: 1; transform: none;             max-height: 4rem; }}
  72%  {{ opacity: 1; transform: none;             max-height: 4rem; }}
  99%  {{ opacity: 0;                              max-height: 4rem; }}
  100% {{ opacity: 0; max-height: 0; padding-top: 0; padding-bottom: 0;
          margin: 0; visibility: hidden; }}
}}
</style>
<div style="background: rgba(33, 195, 84, 0.16); border-radius: 0.5rem;
            padding: 0.7rem 1rem; overflow: hidden;
            animation: {name} {seconds:.2f}s ease forwards;">
  ✓ {text}
</div>
"""
