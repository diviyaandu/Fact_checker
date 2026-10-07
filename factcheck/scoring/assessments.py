"""The LLM's structured per-source judgments, and their validation."""

from dataclasses import dataclass

from factcheck.scoring.constants import RATING_MAX, STANCES


@dataclass
class SourceAssessment:
    source_index: int
    relevance: int      # 0-3
    stance: str         # supports | contradicts | partial | neutral
    directness: int     # 0-3


def _clamp_rating(value) -> int:
    try:
        return max(0, min(RATING_MAX, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def parse_assessments(raw, n_sources: int) -> list[SourceAssessment]:
    """Validates the LLM's source_assessments. Out-of-range or duplicate
    indices are dropped (first wins); bad ratings become 0; unknown stances
    become neutral. Sources the LLM omitted simply contribute nothing."""
    if not isinstance(raw, list):
        return []

    seen: set[int] = set()
    parsed: list[SourceAssessment] = []

    for item in raw:
        if not isinstance(item, dict):
            continue

        idx = item.get("source_index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if not 0 <= idx < n_sources or idx in seen:
            continue

        stance = str(item.get("stance", "neutral")).strip().lower()
        if stance not in STANCES:
            stance = "neutral"

        seen.add(idx)
        parsed.append(
            SourceAssessment(
                source_index=idx,
                relevance=_clamp_rating(item.get("relevance")),
                stance=stance,
                directness=_clamp_rating(item.get("directness")),
            )
        )

    return parsed
