"""
claims.py

Extracts up to 5 checkable factual claims from a transcript using an LLM (Groq/Llama).

Design principle: the LLM never invents timestamps. It must reference numbered
transcript segments, and we look up the real start/end times ourselves. This
prevents timestamp hallucination.
"""

import os
import json
from typing import Literal
from pydantic import BaseModel, ValidationError, field_validator
from groq import Groq
from dotenv import load_dotenv
import time
from logging_utils import logger

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"

ALLOWED_CATEGORIES = ["Science", "Health", "Technology", "History", "General"]


class Claim(BaseModel):
    claim_id: int
    claim_text: str
    start_time: str
    end_time: str
    category: Literal["Science", "Health", "Technology", "History", "General"]
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
        # keep at most 3, and fall back gracefully if the model omitted them
        return v[:3] if v else []


SYSTEM_PROMPT = """You are a claim extraction assistant for a fact-checking tool.

You will be given a numbered transcript from a video. Each line looks like:
[N] text

Your job: identify up to 5 CHECKABLE FACTUAL CLAIMS — statements that assert
something as objectively true or false (a fact, statistic, historical event,
scientific/medical/technical claim).

DO NOT extract:
- Opinions ("I think X is great")
- Greetings, filler, jokes, calls to action ("subscribe now")
- Vague statements with no checkable content
- Questions

For each claim you extract, return:
- "claim_text": the claim, in your own words but staying faithful to exactly
  what was said in the referenced segments. Do not add information that
  wasn't in the transcript.
- "segment_indices": the list of segment numbers (integers) that this claim
  came from. Must be real segment numbers from the transcript you were given.
- "category": one of exactly: "Science", "Health", "Technology", "History", "General"
- "importance": one of exactly: "high", "medium", "low"
- "search_queries": exactly 3 short web search queries that would help verify or
  refute this specific claim. Target the SPECIFIC named entities, places, people,
  or measurements in the claim rather than generic phrasing. If the claim names
  something unusual (an obscure place/lake/person/statistic that may have been
  mis-transcribed from audio), make one query the exact name alone, and one query
  a plausible alternate spelling or transliteration of that name.

Return ONLY a JSON array, no other text, no markdown formatting, no code fences.
Example format:
[
  {
    "claim_text": "...",
    "segment_indices": [2,3],
    "category": "Science",
    "importance": "high",
    "search_queries": ["query one", "query two", "query three"]
  }
]

If there are no checkable factual claims in the transcript, return an empty array: []
Maximum 5 claims total.
"""


def _build_numbered_transcript(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments):
        lines.append(f"[{i}] {seg['text']}")
    return "\n".join(lines)


def _call_llm(numbered_transcript: str) -> str:
    logger.info(
        "GROQ call | stage=claim_extraction | model=%s | input_chars=%d",
        MODEL,
        len(numbered_transcript),
    )

    start = time.time()

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": numbered_transcript},
            ],
            temperature=0.1,
            max_tokens=1500,
        )

        elapsed = time.time() - start

        logger.info(
            "GROQ success | stage=claim_extraction | duration=%.2fs",
            elapsed,
        )

        return response.choices[0].message.content.strip()

    except Exception:
        logger.exception("GROQ failure | stage=claim_extraction")
        raise


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text.strip()


def extract_claims(segments: list[dict]) -> list[Claim]:
    """
    segments: list of {start, end, start_str, end_str, text} from youtube.py/transcription.py
    Returns: list of validated Claim objects, max 5, with real timestamps.
    """
    if not segments:
        return []

    numbered_transcript = _build_numbered_transcript(segments)

    raw_output = _call_llm(numbered_transcript)
    raw_output = _strip_code_fences(raw_output)

    try:
        candidates = json.loads(raw_output)
    except json.JSONDecodeError:
        # one retry with a stricter reminder
        raw_output = _call_llm(
            numbered_transcript + "\n\nREMINDER: Return ONLY valid JSON array, nothing else."
        )
        raw_output = _strip_code_fences(raw_output)
        try:
            candidates = json.loads(raw_output)
        except json.JSONDecodeError:
            return []  # give up cleanly rather than crash the app

    if not isinstance(candidates, list):
        return []

    valid_claims = []
    claim_id = 1

    for c in candidates[:5]:  # hard cap at 5, even if the model returns more
        if not isinstance(c, dict):
            continue

        segment_indices = c.get("segment_indices", [])
        if not segment_indices or not all(isinstance(i, int) for i in segment_indices):
            continue

        # Validate every referenced index actually exists in our transcript
        if any(i < 0 or i >= len(segments) for i in segment_indices):
            continue  # reject claim rather than guess timestamps

        if c.get("category") not in ALLOWED_CATEGORIES:
            continue

        referenced_segments = [segments[i] for i in segment_indices]
        real_start = min(seg["start_str"] for seg in referenced_segments)
        real_end = max(seg["end_str"] for seg in referenced_segments)

        claim_text = c.get("claim_text", "")
        raw_queries = c.get("search_queries", [])
        if not (isinstance(raw_queries, list) and all(isinstance(q, str) and q.strip() for q in raw_queries)):
            raw_queries = []

        # Fallback so search.py always has something usable even if the model
        # skipped this field on a given claim.
        if not raw_queries:
            raw_queries = [
                claim_text,
                f"{claim_text} fact check",
                f"is it true that {claim_text.lower()}",
            ]

        try:
            claim = Claim(
                claim_id=claim_id,
                claim_text=claim_text,
                start_time=real_start,
                end_time=real_end,
                category=c.get("category"),
                importance=c.get("importance", "medium"),
                search_queries=raw_queries,
            )
        except ValidationError:
            continue  # skip malformed claim rather than crash

        valid_claims.append(claim)
        claim_id += 1

    return valid_claims