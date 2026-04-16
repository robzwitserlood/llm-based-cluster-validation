"""
cluster_validator/module.py — DSPy Signature and Module for intruder detection.
"""

import dspy


class IntruderDetectionSignature(dspy.Signature):
    """Identify the one keyword that does not belong with the other five.

    You are given exactly six keywords. Five of them share a common semantic
    theme; one is an intruder from a different domain. Work only with the six
    provided words — do not introduce synonyms or related words that are not
    in the input. In two or three short sentences, state which theme the five
    words share and why the remaining word breaks that theme. Then output that
    word as the intruder.
    """

    keyword_1: str = dspy.InputField(desc="First keyword from the cluster set")
    keyword_2: str = dspy.InputField(desc="Second keyword from the cluster set")
    keyword_3: str = dspy.InputField(desc="Third keyword from the cluster set")
    keyword_4: str = dspy.InputField(desc="Fourth keyword from the cluster set")
    keyword_5: str = dspy.InputField(desc="Fifth keyword from the cluster set")
    keyword_6: str = dspy.InputField(desc="Sixth keyword from the cluster set")

    intruder: str = dspy.OutputField(
        desc=(
            "The single word from the six inputs that does not belong to the same "
            "semantic theme as the other five. Copy the word exactly as given — "
            "no punctuation, no explanation."
        )
    )
    reasoning: str = dspy.OutputField(
        desc=(
            "Two or three sentences explaining which theme the five words share "
            "and why the intruder breaks that theme. Refer only to the six given "
            "words — do not introduce synonyms or outside vocabulary."
        )
    )


class ClusterIntruderValidator(dspy.Module):
    """DSPy module that detects an intruder keyword within a cluster of keywords.

    Uses a Predict step with an explicit reasoning output field declared after
    the intruder field. This ordering matters for small models: because JSON
    adapters serialise fields in declaration order, the model outputs `intruder`
    first (a single word) before expanding on the reasoning. Using
    dspy.ChainOfThought would prepend its own `reasoning` field ahead of
    `intruder`, causing small models to exhaust the token budget on reasoning
    before ever reaching the answer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(IntruderDetectionSignature)

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
