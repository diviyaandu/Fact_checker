"""Evidence retrieval: web search + source-credibility tiers."""

from factcheck.evidence.search import search_evidence_for_claim
from factcheck.evidence.credibility import score_credibility, label_for_tier

__all__ = ["search_evidence_for_claim", "score_credibility", "label_for_tier"]
