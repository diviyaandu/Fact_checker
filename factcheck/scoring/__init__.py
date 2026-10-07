"""Deterministic confidence scoring.

The LLM only produces structured per-source judgments (assessments.py); the
final 0-1 confidence is computed from them by the documented formula in
confidence.py, using the weights/thresholds in constants.py.
"""

from factcheck.scoring.assessments import SourceAssessment, parse_assessments
from factcheck.scoring.confidence import compute_confidence
from factcheck.scoring.constants import (
    CONFLICT_CAP,
    CONFLICT_DOMINANCE_THRESHOLD,
    FORMULA_VERSION,
    MAX_CONFIDENCE,
    TIER_CREDIBILITY,
    WEIGHTS,
)

__all__ = [
    "SourceAssessment",
    "parse_assessments",
    "compute_confidence",
    "CONFLICT_CAP",
    "CONFLICT_DOMINANCE_THRESHOLD",
    "FORMULA_VERSION",
    "MAX_CONFIDENCE",
    "TIER_CREDIBILITY",
    "WEIGHTS",
]
