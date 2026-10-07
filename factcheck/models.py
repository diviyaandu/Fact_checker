"""Shared data models. Field names/shapes are what the disk cache stores, so
changing them invalidates cached results."""

from typing import Literal, get_args

from pydantic import BaseModel, field_validator

from factcheck import config

Category = Literal["Science", "Health", "Technology", "History", "General"]
ALLOWED_CATEGORIES = list(get_args(Category))

VerdictLabel = Literal["TRUE", "FALSE", "PARTIALLY TRUE", "MISLEADING", "UNVERIFIABLE"]
VERDICTS = list(get_args(VerdictLabel))


class Claim(BaseModel):
    claim_id: int
    claim_text: str
    start_time: str
    end_time: str
    category: Category
    importance: Literal["high", "medium", "low"]
    search_queries: list[str] = []

    @field_validator("claim_text")
    @classmethod
    def not_empty(cls, v):
        if not v.strip():
            raise ValueError("claim_text cannot be empty")
        return v.strip()

    @field_validator("search_queries")
    @classmethod
    def cap_queries(cls, v):
        # keep at most MAX_QUERIES_PER_CLAIM, and fall back gracefully if the
        # model omitted them
        return v[: config.MAX_QUERIES_PER_CLAIM] if v else []


class Source(BaseModel):
    title: str
    url: str
    publisher: str
    published_date: str | None = None
    relevant_text: str
    credibility_tier: int = 5


class CitedSource(BaseModel):
    publisher: str
    url: str
    title: str
    relevant_text: str


class Verdict(BaseModel):
    verdict: VerdictLabel
    # Computed deterministically in scoring/ from the LLM's structured
    # per-source assessments — the LLM never supplies this number.
    confidence: float
    summary: str
    explanation: str
    cited_sources: list[CitedSource]
    # Component scores behind `confidence` (None for heuristic/fallback
    # verdicts and for results cached before this field existed).
    confidence_breakdown: dict | None = None
