"""Evidence-grounded verdicts (the LLM judges sources; scoring/ computes confidence)."""

from factcheck.verdicts.checker import BATCH_SIZE, fact_check_claims_batch
from factcheck.verdicts.no_sources import fact_check_no_sources

__all__ = ["BATCH_SIZE", "fact_check_claims_batch", "fact_check_no_sources"]
