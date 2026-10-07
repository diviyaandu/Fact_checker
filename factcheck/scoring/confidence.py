"""
scoring/confidence.py

Deterministic confidence scoring. The LLM (Groq) only produces *structured
per-source judgments* (relevance, stance, directness). This module turns
those judgments plus the existing source-credibility tiers into the final
0-1 confidence. The LLM never supplies the number.

Pure Python, stdlib only.

------------------------------------------------------------------------
FORMULA (confidence = confidence in the verdict reached, NOT P(claim true))
------------------------------------------------------------------------
Per source i (all inputs normalised to 0-1):
    R_i = relevance / 3          LLM rubric 0-3: how directly it addresses the claim
    S_i = directness / 3         LLM rubric 0-3: how explicitly it states the fact
    C_i = TIER_CREDIBILITY[tier] existing Tier 1-5 hierarchy (evidence/credibility.py)
    w_i = R_i * C_i * S_i        evidence "mass" of the source

Only sources with relevance >= 1, directness >= 1 and stance != neutral are
counted. "Aligned" sources are those whose stance matches the verdict:
    TRUE            -> stance "supports"
    FALSE           -> stance "contradicts"
    PARTIALLY TRUE /
    MISLEADING      -> every counted source (mixed evidence is the signal)

Four components, each in [0, 1]:
    relevance   = mean(R_i) over aligned sources
    credibility = R-weighted mean(C_i) over aligned sources
    strength    = dominance * mean(S_i) over aligned sources
                  dominance = aligned mass / (aligned + opposing mass)
                  (TRUE/FALSE: opposing = contradicting/partial for TRUE,
                   supporting/partial for FALSE.  Mixed verdicts: share of
                   mass that is partial or two-sided.)
    agreement   = min(1, independent aligned domains / AGREEMENT_SATURATION)

    raw        = 0.20*relevance + 0.25*credibility + 0.35*strength + 0.20*agreement
    conflict   : TRUE/FALSE with opposing evidence and dominance < 0.75
                 -> raw capped at CONFLICT_CAP (0.60)
    confidence = clamp(raw, 0, MAX_CONFIDENCE=0.95)

Edge cases:
    UNVERIFIABLE                      -> 0.0
    TRUE/FALSE with zero aligned mass -> flagged "no_aligned_evidence"; the
                                         caller downgrades to UNVERIFIABLE (0.0)
    Mixed verdict, zero counted srcs  -> same as above
Weights favour strength of evidence (what the sources actually say) over
retrieval quality; they are heuristic design choices, tunable in one place.
"""

from factcheck.logging_utils import logger
from factcheck.scoring.assessments import SourceAssessment
from factcheck.scoring.constants import (
    AGREEMENT_SATURATION,
    CONFLICT_CAP,
    CONFLICT_DOMINANCE_THRESHOLD,
    DEFAULT_CREDIBILITY,
    FORMULA_VERSION,
    MAX_CONFIDENCE,
    MIN_RATING,
    MIXED_VERDICTS,
    RATING_MAX,
    TIER_CREDIBILITY,
    WEIGHTS,
)


def _domain(source) -> str:
    """Approximate registrable domain, so news.bbc.co.uk and www.bbc.co.uk
    count as ONE independent source. Heuristic (no public-suffix list)."""
    host = (getattr(source, "publisher", "") or "").lower().strip()
    if not host:
        from urllib.parse import urlparse
        host = urlparse(getattr(source, "url", "") or "").netloc.lower()
    host = host.replace("www.", "")

    parts = host.split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in {"co", "com", "org", "gov", "ac", "net"}:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_confidence(verdict: str, assessments: list[SourceAssessment], sources) -> tuple[float, dict]:
    """Returns (confidence, breakdown). breakdown is JSON-serialisable (it is
    cached with the verdict and shown in the UI/logs)."""
    breakdown: dict = {"formula_version": FORMULA_VERSION, "verdict": verdict}

    if verdict == "UNVERIFIABLE":
        breakdown["reason"] = "unverifiable_verdict"
        return 0.0, breakdown

    # --- per-source normalised factors and mass -------------------------
    counted = []
    per_source = []
    for a in assessments:
        src = sources[a.source_index]
        R = a.relevance / RATING_MAX
        S = a.directness / RATING_MAX
        C = TIER_CREDIBILITY.get(src.credibility_tier, DEFAULT_CREDIBILITY)
        w = R * C * S
        used = a.relevance >= MIN_RATING and a.directness >= MIN_RATING and a.stance != "neutral"

        per_source.append({
            "source_index": a.source_index, "domain": _domain(src),
            "tier": src.credibility_tier, "stance": a.stance,
            "R": round(R, 3), "C": round(C, 3), "S": round(S, 3),
            "mass": round(w, 4), "counted": used,
        })
        if used:
            counted.append({"R": R, "C": C, "S": S, "w": w, "stance": a.stance, "domain": _domain(src)})

    breakdown["per_source"] = per_source
    mass = {s: sum(c["w"] for c in counted if c["stance"] == s) for s in ("supports", "contradicts", "partial")}
    breakdown["mass"] = {k: round(v, 4) for k, v in mass.items()}

    # --- choose aligned vs opposing evidence ----------------------------
    mixed = verdict in MIXED_VERDICTS
    if mixed:
        aligned = counted
        mixed_mass = mass["partial"] + 2 * min(mass["supports"], mass["contradicts"])
        total = sum(mass.values())
        dominance = mixed_mass / total if total > 0 else 0.0
        opposing_mass = 0.0
    else:
        aligned_stance = "supports" if verdict == "TRUE" else "contradicts"
        opposite_stance = "contradicts" if verdict == "TRUE" else "supports"
        aligned = [c for c in counted if c["stance"] == aligned_stance]
        aligned_mass = mass[aligned_stance]
        opposing_mass = mass[opposite_stance] + mass["partial"]
        denom = aligned_mass + opposing_mass
        dominance = aligned_mass / denom if denom > 0 else 0.0

    if not aligned or (not mixed and sum(c["w"] for c in aligned) <= 0):
        breakdown["reason"] = "no_aligned_evidence"
        breakdown["no_aligned_evidence"] = True
        return 0.0, breakdown

    # --- four components, each 0-1 --------------------------------------
    r_sum = sum(c["R"] for c in aligned)
    components = {
        "relevance": _safe_mean([c["R"] for c in aligned]),
        "credibility": (sum(c["R"] * c["C"] for c in aligned) / r_sum) if r_sum > 0 else 0.0,
        "strength": dominance * _safe_mean([c["S"] for c in aligned]),
        "agreement": min(1.0, len({c["domain"] for c in aligned}) / AGREEMENT_SATURATION),
    }

    raw = sum(WEIGHTS[k] * components[k] for k in WEIGHTS)
    weighted_sum = raw

    # Conflict handling: a clear verdict with substantial opposing evidence
    # cannot be high-confidence, however credible the supporting sources are.
    conflict_capped = (not mixed) and opposing_mass > 0 and dominance < CONFLICT_DOMINANCE_THRESHOLD
    if conflict_capped:
        raw = min(raw, CONFLICT_CAP)

    confidence = round(max(0.0, min(MAX_CONFIDENCE, raw)), 3)

    breakdown.update({
        "components": {k: round(v, 3) for k, v in components.items()},
        "weights": dict(WEIGHTS),
        "dominance": round(dominance, 3),
        "aligned_sources": len(aligned),
        "independent_domains": len({c["domain"] for c in aligned}),
        "weighted_sum": round(weighted_sum, 3),   # before conflict cap / clamp
        "conflict_capped": conflict_capped,
        "confidence": confidence,
    })

    for row in per_source:
        logger.debug("SCORE source | verdict=%s | %s", verdict, row)

    return confidence, breakdown
