"""
cluster_validator/module.py — DSPy Signature and Module for intruder detection.
"""

import dspy


class IntruderDetectionSignature(dspy.Signature):
    """Identify the intruder keyword in a set of cluster keywords.

    Given six keywords where five belong to a coherent text cluster and one
    is an intruder that does not fit, return step-by-step reasoning followed
    by the single word that is the intruder.
    """

    keyword_1: str = dspy.InputField(desc="First keyword from the cluster set")
    keyword_2: str = dspy.InputField(desc="Second keyword from the cluster set")
    keyword_3: str = dspy.InputField(desc="Third keyword from the cluster set")
    keyword_4: str = dspy.InputField(desc="Fourth keyword from the cluster set")
    keyword_5: str = dspy.InputField(desc="Fifth keyword from the cluster set")
    keyword_6: str = dspy.InputField(desc="Sixth keyword from the cluster set")

    reasoning: str = dspy.OutputField(
        desc=(
            "Step-by-step explanation of the semantic relationships between the six keywords "
            "and why one of them does not belong to the same cluster as the other five."
        )
    )
    intruder: str = dspy.OutputField(
        desc=(
            "The single keyword that does not belong to the same semantic cluster "
            "as the other five. Return only the word itself, nothing else."
        )
    )


class ClusterIntruderValidator(dspy.Module):
    """DSPy module that detects an intruder keyword within a cluster of keywords.

    Uses Chain-of-Thought reasoning to analyse the semantic relationships
    between six keywords and identify the one that does not belong.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.ChainOfThought(IntruderDetectionSignature)

    def forward(
        self,
        keyword_1: str,
        keyword_2: str,
        keyword_3: str,
        keyword_4: str,
        keyword_5: str,
        keyword_6: str,
    ) -> dspy.Prediction:
        return self.predictor(
            keyword_1=keyword_1,
            keyword_2=keyword_2,
            keyword_3=keyword_3,
            keyword_4=keyword_4,
            keyword_5=keyword_5,
            keyword_6=keyword_6,
        )


def find_intruder(keywords: list[str]) -> str:
    """Find the intruder keyword in a list of exactly six keywords.

    Assumes DSPy has already been configured via configure_dspy().

    Args:
        keywords: A list of exactly six keyword strings.

    Returns:
        The keyword identified as the intruder.

    Raises:
        ValueError: If the keyword list does not contain exactly six items.
    """
    if len(keywords) != 6:
        raise ValueError(f"Expected exactly 6 keywords, got {len(keywords)}.")

    validator = ClusterIntruderValidator()
    result = validator(
        keyword_1=keywords[0],
        keyword_2=keywords[1],
        keyword_3=keywords[2],
        keyword_4=keywords[3],
        keyword_5=keywords[4],
        keyword_6=keywords[5],
    )
    return result.intruder.strip()
