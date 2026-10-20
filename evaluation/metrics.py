"""Pure classification metrics (stdlib only, no API/network imports).

Ground truth is only ever read here; nothing in this module can change it.
"""

LABELS = ["TRUE", "FALSE", "UNVERIFIABLE"]


def confusion_matrix(y_true: list[str], y_pred: list[str], labels: list[str] = LABELS) -> list[list[int]]:
    """matrix[i][j] = claims whose truth is labels[i] and prediction is labels[j]."""
    idx = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1
    return matrix


def _div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(y_true: list[str], y_pred: list[str], labels: list[str] = LABELS) -> dict:
    """Per-class precision/recall/F1, macro F1 (primary), accuracy (secondary).

    A class with no predictions has precision 0; with no true items, recall 0.
    Macro F1 is the unweighted mean of the per-class F1s over ALL `labels`
    (classes are never dropped).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred differ in length")
    if not y_true:
        raise ValueError("no samples to evaluate")

    matrix = confusion_matrix(y_true, y_pred, labels)
    per_class = {}

    for i, label in enumerate(labels):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(labels))) - tp
        fn = sum(matrix[i]) - tp
        precision = _div(tp, tp + fp)
        recall = _div(tp, tp + fn)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": _div(2 * precision * recall, precision + recall),
            "support": tp + fn,
        }

    correct = sum(matrix[i][i] for i in range(len(labels)))

    return {
        "n": len(y_true),
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(labels),
        "accuracy": correct / len(y_true),
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": matrix,  # rows = ground truth, columns = prediction
    }
