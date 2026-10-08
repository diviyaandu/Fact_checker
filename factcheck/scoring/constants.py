"""Scoring constants — the weights and thresholds of the confidence formula
(see confidence.py for how they are used)."""

FORMULA_VERSION = "v2"   # v2 = v1 + NLI adjustment (no-op when NLI is unavailable)

# Tier 1 (government/official) ... Tier 5 (unrated). Mirrors evidence/credibility.py.
TIER_CREDIBILITY = {1: 1.00, 2: 0.85, 3: 0.65, 4: 0.45, 5: 0.25}
DEFAULT_CREDIBILITY = TIER_CREDIBILITY[5]

WEIGHTS = {
    "relevance": 0.20,
    "credibility": 0.25,
    "strength": 0.35,
    "agreement": 0.20,
}

MIN_RATING = 1                     # relevance/directness below this => source ignored
RATING_MAX = 3                     # LLM rubric scale is 0-3
AGREEMENT_SATURATION = 3           # 3 independent domains = full agreement credit
CONFLICT_DOMINANCE_THRESHOLD = 0.75
CONFLICT_CAP = 0.60
MAX_CONFIDENCE = 0.95              # never claim certainty

STANCES = {"supports", "contradicts", "partial", "neutral"}
MIXED_VERDICTS = {"PARTIALLY TRUE", "MISLEADING"}

# ---- NLI second opinion (see scoring/nli.py) --------------------------------
# Groq stance -> 3-way NLI label. "partial" is inherently in-between, so it maps
# to neutral and carries no NLI agreement signal.
STANCE_TO_NLI = {
    "supports": "entailment",
    "contradicts": "contradiction",
    "partial": "neutral",
    "neutral": "neutral",
}
NLI_DEADZONE = 0.20        # |per-source agreement| below this => 0 (ambiguous NLI)
NLI_MAX_BOOST = 0.08       # confidence added when NLI fully agrees with Groq
NLI_MAX_PENALTY = 0.15     # confidence removed when NLI fully disagrees (asymmetric: disagreement hurts more)
