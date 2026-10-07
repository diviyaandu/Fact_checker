"""Scoring constants — the weights and thresholds of the confidence formula
(see confidence.py for how they are used)."""

FORMULA_VERSION = "v1"

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
