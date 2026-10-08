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

# ---- NLI second opinion (local, no API call) ---------------------------------
# NOTE: there is no official "microsoft/deberta-v3-base-mnli" checkpoint; this is
# the DeBERTa-v3-base MNLI(+FEVER+ANLI) checkpoint. Label order is read from the
# model config, so any 3-way MNLI model can be swapped in via NLI_MODEL.
NLI_ENABLED = os.getenv("NLI_ENABLED", "1").lower() not in ("0", "false", "no")
NLI_MODEL = os.getenv("NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
NLI_DEVICE = os.getenv("NLI_DEVICE", "cpu")
NLI_BATCH_SIZE = 8                  # (claim, snippet) pairs per forward pass
NLI_MAX_LENGTH = 512                # tokens; the snippet (premise) is truncated first

# ---- Transcription ---------------------------------------------------------
WHISPER_MODEL_SIZE = "tiny"
