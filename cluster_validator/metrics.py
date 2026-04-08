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
    gold = example.intruder.strip().lower()
    predicted = pred.intruder.strip().lower()
    return gold == predicted or gold in predicted.split()


def gepa_metric(gold, pred, trace, pred_name, pred_trace) -> bool:
    """5-argument metric wrapper required by the GEPA optimizer."""
    return intruder_exact_match(gold, pred, trace)
