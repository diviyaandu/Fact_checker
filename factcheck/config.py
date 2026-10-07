"""Backend settings: environment variables and tunable constants in one place.

(Scoring weights live next to the formula in scoring/constants.py; UI settings
live in ui/config.py.)
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ---- API keys --------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ---- LLM models ------------------------------------------------------------
# Claim extraction has always used a fixed model; verdicts can be switched with
# the GROQ_MODEL environment variable. Both default to the same model.
CLAIMS_MODEL = "openai/gpt-oss-120b"
VERDICT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ---- Claims ----------------------------------------------------------------
MAX_CLAIMS = 5                      # hard cap, even if the model returns more
MAX_QUERIES_PER_CLAIM = 3

# ---- Evidence search (Tavily) ----------------------------------------------
MAX_RESULTS_PER_QUERY = 3
MAX_SOURCES_PER_CLAIM = 5
MAX_CONCURRENT_SEARCHES = 2         # conservative for free-tier rate limits
MAX_SNIPPET_CHARS = 800

# ---- Verdicts --------------------------------------------------------------
BATCH_SIZE = 2                      # claims per Groq fact-check call

# ---- Transcription ---------------------------------------------------------
WHISPER_MODEL_SIZE = "tiny"
