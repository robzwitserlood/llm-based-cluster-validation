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

    trefwoord_1: str = dspy.InputField(desc="Eerste trefwoord")
    trefwoord_2: str = dspy.InputField(desc="Tweede trefwoord")
    trefwoord_3: str = dspy.InputField(desc="Derde trefwoord")
    trefwoord_4: str = dspy.InputField(desc="Vierde trefwoord")
    trefwoord_5: str = dspy.InputField(desc="Vijfde trefwoord")
    trefwoord_6: str = dspy.InputField(desc="Zesde trefwoord")

    indringer: str = dspy.OutputField(
        desc=(
            "Het ene trefwoord dat niet tot hetzelfde concept behoort als de overige "
            "vijf trefwoorden. Geef een exacte kopie van het trefwoordwoord."
        )
    )
    redenering: str = dspy.OutputField(
        desc=(
            "Een korte uitleg van het concept waartoe de vijf gerelateerde trefwoorden "
            "behoren en waarom de indringer daarbuiten valt."
        )
    )


class ClusterIntruderValidator(dspy.Module):
    """DSPy module that detects an intruder keyword within a cluster of keywords.

    Uses a Predict step with an explicit redenering output field declared after
    the indringer field. This ordering matters for small models: because JSON
    adapters serialise fields in declaration order, the model outputs `indringer`
    first (a single word) before expanding on the redenering. Using
    dspy.ChainOfThought would prepend its own `reasoning` field ahead of
    `indringer`, causing small models to exhaust the token budget on redenering
    before ever reaching the answer.
    """

    def __init__(self) -> None:
        super().__init__()
        self.predictor = dspy.Predict(IntruderDetectionSignature)

    def forward(
        self,
        trefwoord_1: str,
        trefwoord_2: str,
        trefwoord_3: str,
        trefwoord_4: str,
        trefwoord_5: str,
        trefwoord_6: str,
    ) -> dspy.Prediction:
        return self.predictor(
            trefwoord_1=trefwoord_1,
            trefwoord_2=trefwoord_2,
            trefwoord_3=trefwoord_3,
            trefwoord_4=trefwoord_4,
            trefwoord_5=trefwoord_5,
            trefwoord_6=trefwoord_6,
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
        trefwoord_1=keywords[0],
        trefwoord_2=keywords[1],
        trefwoord_3=keywords[2],
        trefwoord_4=keywords[3],
        trefwoord_5=keywords[4],
        trefwoord_6=keywords[5],
    )
    return result.indringer.strip()
