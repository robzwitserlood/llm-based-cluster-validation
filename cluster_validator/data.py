"""
cluster_validator/data.py — dataset loading and splitting for ClusterIntruderValidator.

Provides build_devset() which returns a list of dspy.Example objects, and a
two-level splitting API:

  1. split_test()          — carve out a locked held-out test set (same across all optimizers)
  2. split_for_bootstrap() — 20% train / 80% dev  (prompt optimizer convention per DSPy docs)
     split_for_gepa()      — 70% train / 15% val / 15% dev  (GEPA: maximize train)
     split_for_finetune()  — 80% train / 20% dev  (weight optimizer: maximize train)

Always call split_test() first with the same seed so the test set is identical
across optimizer runs, enabling direct accuracy comparisons.
"""

import random
import json
from pathlib import Path

import dspy

EXAMPLES_FILE = Path(__file__).parent.parent / "data" / "raw_examples.json"

SPLIT_SEED = 42  # shared constant — must be the same across all optimizer runs

# Hand-crafted fallback examples — (k1, k2, k3, k4, k5, k6, indringer)
_FALLBACK_EXAMPLES: list[tuple[str, ...]] = [
    ("ocean",    "river",          "lake",           "mountain",  "stream",    "pond",        "mountain"),
    ("lion",     "tiger",          "cheetah",        "leopard",   "banana",    "jaguar",      "banana"),
    ("python",   "java",           "rust",           "carrot",    "go",        "kotlin",      "carrot"),
    ("apple",    "mango",          "grape",          "peach",     "plum",      "hammer",      "hammer"),
    ("mercury",  "venus",          "earth",          "mars",      "jupiter",   "saxophone",   "saxophone"),
    ("guitar",   "piano",          "violin",         "trumpet",   "stapler",   "cello",       "stapler"),
    ("red",      "blue",           "green",          "yellow",    "purple",    "nitrogen",    "nitrogen"),
    ("carrot",   "broccoli",       "spinach",        "kale",      "zucchini",  "monday",      "monday"),
    ("iron",     "gold",           "silver",         "copper",    "sandcastle","platinum",    "sandcastle"),
    ("rain",     "snow",           "hail",           "fog",       "thunder",   "algebra",     "algebra"),
    ("france",   "germany",        "italy",          "spain",     "sweden",    "kenya",       "kenya"),
    ("football", "tennis",         "swimming",       "cycling",   "boxing",    "sourdough",   "sourdough"),
    ("knife",    "fork",           "spoon",          "ladle",     "whisk",     "electron",    "electron"),
    ("oak",      "pine",           "maple",          "birch",     "sequoia",   "trumpet",     "trumpet"),
    ("joy",      "sadness",        "anger",          "fear",      "disgust",   "nitrogen",    "nitrogen"),
    ("dollar",   "euro",           "yen",            "pound",     "rupee",     "volcano",     "volcano"),
    ("eagle",    "sparrow",        "parrot",         "flamingo",  "raven",     "suitcase",    "suitcase"),
    ("loop",     "function",       "variable",       "class",     "module",    "umbrella",    "umbrella"),
    ("star",     "galaxy",         "nebula",         "comet",     "asteroid",  "cucumber",    "cucumber"),
    ("pen",      "stapler",        "notebook",       "scissors",  "ruler",     "avalanche",   "avalanche"),
    ("silk",     "cotton",         "linen",          "wool",      "polyester", "asteroid",    "asteroid"),
    ("coffee",   "tea",            "juice",          "milk",      "water",     "parliament",  "parliament"),
    ("flu",      "cold",           "measles",        "typhoid",   "malaria",   "harmonica",   "harmonica"),
    ("everest",  "k2",             "kangchenjunga",  "lhotse",    "makalu",    "keyboard",    "keyboard"),
    ("atlantic", "pacific",        "indian",         "arctic",    "southern",  "bicycle",     "bicycle"),
]

_INPUT_FIELDS = ("trefwoord_1", "trefwoord_2", "trefwoord_3", "trefwoord_4", "trefwoord_5", "trefwoord_6")


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_test(
    examples: list[dspy.Example],
    test_frac: float = 0.15,
    seed: int = SPLIT_SEED,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Carve out a locked held-out test set.

    Call this first, before any optimizer-specific split. Use the same
    seed and test_frac across all runs to guarantee an identical test set.

    Returns:
        (trainable, test) where trainable is passed to split_for_*().
    """
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * (1 - test_frac)))
    return shuffled[:cut], shuffled[cut:]


def split_for_bootstrap(
    trainable: list[dspy.Example],
    seed: int = SPLIT_SEED,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """20% train / 80% dev — DSPy recommended split for prompt optimizers.

    Prompt optimizers (BootstrapFewShot) overfit to small training sets,
    so DSPy recommends an inverted split that prioritises stable validation.

    Returns:
        (train, dev)
    """
    rng = random.Random(seed)
    shuffled = trainable[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * 0.20))
    return shuffled[:cut], shuffled[cut:]


def split_for_gepa(
    trainable: list[dspy.Example],
    seed: int = SPLIT_SEED,
) -> tuple[list[dspy.Example], list[dspy.Example], list[dspy.Example]]:
    """70% train / 15% val / 15% dev — GEPA convention: maximise train size.

    GEPA uses trainset for reflective updates and valset for Pareto scoring.

    Returns:
        (train, val, dev)
    """
    rng = random.Random(seed)
    shuffled = trainable[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    cut1 = max(1, int(n * 0.70))
    cut2 = max(cut1 + 1, int(n * 0.85))
    return shuffled[:cut1], shuffled[cut1:cut2], shuffled[cut2:]


def split_for_finetune(
    trainable: list[dspy.Example],
    seed: int = SPLIT_SEED,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """80% train / 20% dev — standard ML split for weight-level optimizers.

    BootstrapFinetune needs enough successful traces to fine-tune on,
    so a large training set is preferred.

    Returns:
        (train, dev)
    """
    rng = random.Random(seed)
    shuffled = trainable[:]
    rng.shuffle(shuffled)
    cut = max(1, int(len(shuffled) * 0.80))
    return shuffled[:cut], shuffled[cut:]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def build_devset(examples_file: Path | str = EXAMPLES_FILE) -> list[dspy.Example]:
    """Return a devset of dspy.Example objects.

    Loads from *examples_file* (JSON produced by pipeline/build_dataset.py) when
    it exists; otherwise falls back to the built-in hand-crafted examples.
    """
    path = Path(examples_file)
    if path.exists():
        print(f"Loading examples from {path} …")
        with open(path) as f:
            rows = json.load(f)
        devset = [
            dspy.Example(
                trefwoord_1=r["trefwoord_1"],
                trefwoord_2=r["trefwoord_2"],
                trefwoord_3=r["trefwoord_3"],
                trefwoord_4=r["trefwoord_4"],
                trefwoord_5=r["trefwoord_5"],
                trefwoord_6=r["trefwoord_6"],
                indringer=r["indringer"],
            ).with_inputs(*_INPUT_FIELDS)
            for r in rows
        ]
        print(f"  {len(devset)} examples loaded.")
        return devset

    print(f"{path} not found — using built-in fallback examples.")
    print("Run `python -m pipeline.build_dataset` to generate real-world examples.")
    return [
        dspy.Example(
            trefwoord_1=k1, trefwoord_2=k2, trefwoord_3=k3,
            trefwoord_4=k4, trefwoord_5=k5, trefwoord_6=k6,
            indringer=indringer,
        ).with_inputs(*_INPUT_FIELDS)
        for k1, k2, k3, k4, k5, k6, indringer in _FALLBACK_EXAMPLES
    ]
