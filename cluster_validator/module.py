"""
cluster_validator/module.py — DSPy Signature and Module for intruder detection.
"""

import dspy


class IntruderDetectionSignature(dspy.Signature):
    """Identificeer het trefwoord dat niet bij de andere vijf hoort.
    
    Je krijgt exact zes trefwoorden. Vijf daarvan delen een gemeenschappelijk 
    semantisch concept; één is een indringer die conceptueel anders is. In twee 
    of drie korte zinnen, geef aan welk concept de vijf woorden delen en waarom 
    het overgebleven woord niet binnen dit concept past. Geef vervolgens het 
    overgebleven woord als de indringer.
    """

    keyword_1: str = dspy.InputField(desc="Eerste trefwoord")
    keyword_2: str = dspy.InputField(desc="Tweede trefwoord")
    keyword_3: str = dspy.InputField(desc="Derde trefwoord")
    keyword_4: str = dspy.InputField(desc="Vierde trefwoord")
    keyword_5: str = dspy.InputField(desc="Vijfde trefwoord")
    keyword_6: str = dspy.InputField(desc="Zesde trefwoord")

    intruder: str = dspy.OutputField(
        desc=(
            "Het ene trefwoord dat niet tot hetzelfde concept behoort als de overige "
            "vijf trefwoorden. Geef een exacte kopie van het trefwoordwoord."
        )
    )
    reasoning: str = dspy.OutputField(
        desc=(
            "Een korte uitleg van het concept waartoe de vijf gerelateerde trefwoorden "
            "behoren en waarom de indringer daarbuiten valt."
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
