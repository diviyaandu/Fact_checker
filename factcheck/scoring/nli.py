"""
Local NLI second opinion on the LLM's per-source stance.

For each (claim, source snippet) pair a local MNLI model (DeBERTa) predicts
entailment / neutral / contradiction. Its agreement with the Groq stance feeds
a bounded adjustment in confidence.py. No API call; torch/transformers are
imported lazily, and any failure degrades to "no NLI" (existing score unchanged).

Premise = snippet, hypothesis = claim.

Per-source agreement a_i in [-1, 1]:
    supports    -> P(entailment)    - P(contradiction)
    contradicts -> P(contradiction) - P(entailment)
    partial / neutral -> 0   (maps to "neutral"; no signal)
    |a_i| < NLI_DEADZONE -> 0  (neutral/ambiguous NLI gives no boost)
"""

from dataclasses import dataclass, field
from functools import lru_cache

from factcheck import config
from factcheck.logging_utils import logger
from factcheck.scoring.constants import MIN_RATING, NLI_DEADZONE, STANCE_TO_NLI

LABELS = ("entailment", "neutral", "contradiction")


@dataclass
class NLIResult:
    label: str                                   # argmax of probs
    probs: dict = field(default_factory=dict)    # {"entailment","neutral","contradiction"} -> float

    def to_dict(self) -> dict:
        return {"label": self.label, "probs": {k: round(v, 4) for k, v in self.probs.items()}}


def result_from_probs(probs: dict) -> NLIResult:
    return NLIResult(label=max(LABELS, key=lambda k: probs.get(k, 0.0)), probs=dict(probs))


# ---- pure scoring helpers (no torch) ----------------------------------------

def stance_to_nli_label(stance: str) -> str:
    return STANCE_TO_NLI.get(stance, "neutral")


def source_agreement(stance: str, result: NLIResult) -> float:
    p = result.probs
    if stance == "supports":
        a = p.get("entailment", 0.0) - p.get("contradiction", 0.0)
    elif stance == "contradicts":
        a = p.get("contradiction", 0.0) - p.get("entailment", 0.0)
    else:
        return 0.0
    return 0.0 if abs(a) < NLI_DEADZONE else a


def counted_indices(assessments) -> list[int]:
    """Sources the confidence formula counts (same filter as confidence.py)."""
    return [
        a.source_index for a in assessments
        if a.relevance >= MIN_RATING and a.directness >= MIN_RATING and a.stance != "neutral"
    ]


# ---- model (lazy, cached) ----------------------------------------------------

@lru_cache(maxsize=1)
def _load_model():
    import torch  # noqa: F401
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("NLI | loading model=%s device=%s", config.NLI_MODEL, config.NLI_DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL)
    model.to(config.NLI_DEVICE).eval()

    # Label order differs between checkpoints; read it from the config.
    order = []
    for i in range(model.config.num_labels):
        name = str(model.config.id2label[i]).lower()
        order.append(next(l for l in LABELS if name.startswith(l[:5])))
    return tokenizer, model, order


def _predict(pairs: list[tuple[str, str]]) -> list[dict]:
    """pairs: (premise, hypothesis). One forward pass per NLI_BATCH_SIZE chunk."""
    import torch

    tokenizer, model, order = _load_model()
    out = []
    for i in range(0, len(pairs), config.NLI_BATCH_SIZE):
        chunk = pairs[i:i + config.NLI_BATCH_SIZE]
        enc = tokenizer(
            [p for p, _ in chunk], [h for _, h in chunk],
            truncation="only_first", max_length=config.NLI_MAX_LENGTH,
            padding=True, return_tensors="pt",
        ).to(config.NLI_DEVICE)
        with torch.inference_mode():
            probs = torch.softmax(model(**enc).logits, dim=-1).tolist()
        out.extend({lab: row[j] for j, lab in enumerate(order)} for row in probs)
    return out


def classify_pairs(pairs: list[tuple[str, str]]) -> list[NLIResult]:
    """(snippet, claim) pairs -> NLIResult list. [] if NLI is disabled/unavailable."""
    if not pairs or not config.NLI_ENABLED:
        return []
    try:
        return [result_from_probs(p) for p in _predict(pairs)]
    except Exception as exc:  # missing deps, download failure, OOM ...
        logger.warning("NLI | unavailable, skipping | %s: %s", type(exc).__name__, exc)
        return []


def nli_for_claims(requests) -> list[dict[int, NLIResult]]:
    """requests: [(claim_text, sources, source_indices)] -> per-request {source_index: NLIResult}.
    All pairs from all claims go through ONE batched call."""
    flat = [(sources[i].relevant_text, claim, k, i)
            for k, (claim, sources, idxs) in enumerate(requests) for i in idxs]
    results = classify_pairs([(p, h) for p, h, _, _ in flat])
    out: list[dict[int, NLIResult]] = [{} for _ in requests]
    if len(results) == len(flat):
        for (_, _, k, i), r in zip(flat, results):
            out[k][i] = r
    return out
