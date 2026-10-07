"""
cache.py

Minimal disk cache, no extra dependencies. Two purposes:

1. Development/demo speed: re-running the same video doesn't re-burn
   Groq/Tavily calls.
2. Evaluation validity: your baseline-vs-improved comparison should be
   run against the SAME claims/evidence each time you re-test a stage,
   otherwise LLM non-determinism confounds your before/after numbers.
   Caching intermediate results (transcript, claims) lets you swap out
   just the verdict logic and compare fairly.

Not thread-safe by design — fine for a single Streamlit session.
Cache lives at <project root>/.cache/fact_checker_cache.json.
"""

import os
import json
import hashlib
import threading

# Project root (one level above this package), so the cache stays at
# <project>/.cache/fact_checker_cache.json exactly as before the refactor.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_PATH = os.path.join(_PROJECT_ROOT, ".cache", "fact_checker_cache.json")
_lock = threading.Lock()


def _load() -> dict:
    if not os.path.exists(_CACHE_PATH):
        return {}
    try:
        with open(_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    tmp_path = _CACHE_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp_path, _CACHE_PATH)  # atomic on POSIX + Windows


def make_key(*parts: str) -> str:
    """Stable short key from one or more strings (e.g. video_id, claim_text)."""
    joined = "||".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:24]


def get(namespace: str, key: str):
    """Returns the cached value, or None if absent/corrupt."""
    with _lock:
        data = _load()
    return data.get(namespace, {}).get(key)


def set(namespace: str, key: str, value) -> None:
    """value must be JSON-serializable (plain dicts/lists/strings/numbers)."""
    with _lock:
        data = _load()
        data.setdefault(namespace, {})[key] = value
        _save(data)


def clear() -> None:
    with _lock:
        if os.path.exists(_CACHE_PATH):
            os.remove(_CACHE_PATH)
