"""
cluster_validator/metrics.py — evaluation metrics for ClusterIntruderValidator.
"""

import dspy


def intruder_exact_match(example: dspy.Example, pred: dspy.Prediction, trace=None) -> bool:
    """Return True if the predicted intruder matches the gold label.

    Case-insensitive. Accepts the gold label anywhere in the prediction to
    handle cases where the model returns e.g. "mountain." or
    "The intruder is mountain".
    """
    gold = example.indringer.strip().lower()
    raw = getattr(pred, "indringer", None)
    if raw is None:
        return False
    predicted = raw.strip().lower()
    return gold == predicted or gold in predicted.split()


def gepa_metric(gold, pred, trace, pred_name=None, pred_trace=None) -> bool:
    """5-argument metric wrapper required by the GEPA optimizer."""
    try:
        return intruder_exact_match(gold, pred, trace)
    except Exception:
        return False
