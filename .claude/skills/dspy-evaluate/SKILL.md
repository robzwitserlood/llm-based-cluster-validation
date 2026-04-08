---
name: dspy-evaluate
description: Evaluate a DSPy module against a dataset with a metric. Use when the user wants to measure program quality, write a metric function, build a dev/test dataset, run dspy.Evaluate, or establish a baseline before optimization.
argument-hint: [module name or task description]
allowed-tools: Read Grep Glob
---

You are setting up evaluation for a DSPy module. Target: `$ARGUMENTS` (if blank, find the main Module in the project).

## Step 1 — Read the existing codebase

1. Glob `**/*.py` and read every file containing `import dspy` or `from dspy`
2. Identify the Module and its Signature's InputField / OutputField names
3. Check for any existing `dspy.Example` objects, metric functions, or dataset files

## Step 2 — Build the dataset

DSPy uses `dspy.Example` objects — dict-like with dot-access and an `.with_inputs()` marker.

```python
# One example per row; with_inputs() marks which fields are inputs vs gold labels
example = dspy.Example(
    question="What is the capital of France?",
    answer="Paris"
).with_inputs("question")

# Access subsets
example.inputs()   # → Example(question=...)
example.labels()   # → Example(answer=...)
```

Rules:
- **`.with_inputs()` is required** — it tells DSPy which fields the program receives; omitting it is the most common mistake
- Aim for ≥ 20 examples for a meaningful eval; ≥ 200 for optimization
- Split into `devset` (eval during dev) and optionally `testset` (held out until the end)
- Load from HuggingFace with `datasets` or build manually; save as JSON to avoid re-loading

## Step 3 — Write the metric function

Signature: `def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float | bool`

| Output type | When to use |
|-------------|-------------|
| `bool` | Hard pass/fail — classification accuracy, exact match |
| `float` in [0, 1] | Graded quality — partial credit, blended scores |

**Simple metrics**

```python
# Exact match (case-insensitive)
def metric(example, pred, trace=None):
    return pred.answer.strip().lower() == example.answer.strip().lower()

# Built-ins for QA
from dspy.evaluate import answer_exact_match, answer_passage_match
```

**LM-judge metric (for open-ended outputs)**

```python
class Faithfulness(dspy.Signature):
    """Is the prediction faithful to the context? Answer yes or no."""
    context: str = dspy.InputField()
    prediction: str = dspy.InputField()
    faithful: bool = dspy.OutputField()

judge = dspy.Predict(Faithfulness)

def metric(example, pred, trace=None):
    result = judge(context=example.context, prediction=pred.summary)
    return result.faithful
```

**Blended metric**

```python
def metric(example, pred, trace=None):
    correct = pred.answer.strip().lower() == example.answer.strip().lower()
    concise = len(pred.answer.split()) <= 20
    return 0.7 * correct + 0.3 * concise
```

**Trace-aware metric** (used during optimization to inspect intermediate steps)

```python
def metric(example, pred, trace=None):
    score = pred.answer.lower() == example.answer.lower()
    if trace is not None:
        # Extra strictness during optimizer compilation
        score = score and len(pred.reasoning.split()) < 100
    return score
```

Keep the metric fast — it runs once per example per optimizer trial.

## Step 4 — Run dspy.Evaluate

```python
import dspy

dspy.configure(lm=dspy.LM(..., cache=True))  # cache=True avoids re-calling identical inputs

devset = [
    dspy.Example(question=..., answer=...).with_inputs("question"),
    # ...
]

def metric(example, pred, trace=None):
    return pred.answer.strip().lower() == example.answer.strip().lower()

program = MyModule()

evaluate = dspy.Evaluate(
    devset=devset,
    metric=metric,
    num_threads=4,        # parallelism
    display_progress=True,
    display_table=5,      # show first 5 rows of results
)

score = evaluate(program)
print(f"Accuracy: {score:.1f}%")
```

## Step 5 — Interpret results and iterate

- **Examine failures first** — set `display_table=20` or iterate `devset` manually to see where the program goes wrong
- **Check intermediate outputs** — use `dspy.inspect_history(n=5)` to see what prompts and responses were generated
- **Establish a baseline score** before optimizing — you need a number to beat
- If the metric is also a DSPy program (LM judge), you can optimize the metric itself using labeled metric examples

## DSPy idioms

- `dspy.LM(..., cache=True)` is essential to avoid re-running identical calls
- Start with a simple metric (exact match, regex) before adding an LM judge — simpler is faster and more debuggable
- `dspy.Evaluate` returns a float score (percentage of examples where metric returned truthy)
- The `trace` parameter in the metric signature is `None` during evaluation and populated during optimizer compilation — always include it even if unused
- Use `dspy.Assert` / `dspy.Suggest` inside the Module (not the metric) to add constrained retry logic
