"""
Test helper: provide inert stand-ins for heavy third-party SDKs that aren't
installed (groq, tavily, streamlit, ...). Everything that would touch the
network/GPU/UI is mocked in the tests, so the stand-ins are never really used.
Installed packages are left alone.

Usage (at the top of a test module):  import _stubs; _stubs.install()
"""

import os
import sys
from types import SimpleNamespace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_HEAVY = ("groq", "tavily", "youtube_transcript_api", "yt_dlp", "faster_whisper", "streamlit")


def install() -> None:
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    os.environ.setdefault("GROQ_API_KEY", "test-key")
    os.environ.setdefault("TAVILY_API_KEY", "test-key")

    for name in _HEAVY:
        try:
            __import__(name)
        except ImportError:
            stub = type(sys)(name)
            stub.__getattr__ = lambda attr: (lambda *a, **k: SimpleNamespace())
            sys.modules[name] = stub
