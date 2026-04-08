---
name: dspy-optimize
description: Set up a DSPy optimization loop for an existing Module. Use when the user wants to improve a DSPy program using an optimizer (BootstrapFewShot, MIPROv2, COPRO, BootstrapFinetune), define a metric, create training examples, or evaluate program quality.
argument-hint: [module name or description]
allowed-tools: Read Grep Glob
---

You are setting up a DSPy optimization loop. Target: `$ARGUMENTS` (if blank, find the main Module in the project).

## Step 1 — Read the existing codebase

1. Glob `**/*.py` and read every file containing `import dspy` or `from dspy`
2. Identify the Module to optimize and its Signature's InputField / OutputField names
3. Check for any existing `dspy.Example` objects, metric functions, or saved `.json` program files

## Step 2 — Scaffold training examples

- Each example: `dspy.Example(input_field=..., output_field=...).with_inputs("input_field")`
- **`.with_inputs()` is required** — it tells DSPy which fields are inputs vs gold labels; omitting it is the most common mistake
- Provide at least 5–10 representative stub examples; note where the user should fill in real data

## Step 3 — Write the metric function

Signature: `def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float | bool`

- Return `bool` for pass/fail; return `float` in [0, 1] for graded quality
- Access fields: `example.field_name` for gold labels, `pred.field_name` for predictions
- Keep the metric fast — it runs once per example per optimizer trial
- For exact match: `return pred.output.strip().lower() == example.output.strip().lower()`
- For semantic similarity: use an embedding or an LM judge inside the metric

## Step 4 — Choose the optimizer

| Situation | Optimizer |
|-----------|-----------|
| < 50 labelled examples, want few-shot demos | `dspy.BootstrapFewShot(metric=metric)` |
| 50–300 examples, want best quality | `dspy.MIPROv2(metric=metric, auto="medium")` |
| Want to tune instructions too | `dspy.MIPROv2(metric=metric, auto="heavy")` |
| Want to finetune LM weights | `dspy.BootstrapFinetune(metric=metric)` |

Always set `max_bootstrapped_demos` and `max_labeled_demos` explicitly for reproducibility.

## Step 5 — Produce the full optimization script

```python
import dspy

# Configure LM (read from dspy_config.yaml if it exists)
dspy.configure(lm=dspy.LM(..., cache=True))  # cache=True avoids re-running calls during optimization

# Training data
trainset = [
    dspy.Example(input_field=..., label=...).with_inputs("input_field"),
    # ...
]
devset = [...]  # held-out examples for evaluation

# Metric
def metric(example, pred, trace=None):
    ...

# Evaluate baseline (before optimization)
evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
program = MyModule()
print("Baseline:", evaluate(program))

# Optimize
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=16)
compiled_program = optimizer.compile(program, trainset=trainset)

# Evaluate after
print("Optimized:", evaluate(compiled_program))

# Save — always save so the result is not lost
compiled_program.save("program_optimized.json")
```

## Step 6 — Show how to reload

```python
program = MyModule()
program.load("program_optimized.json")
result = program(input_field=...)
```

## DSPy idioms

- `dspy.LM(..., cache=True)` is essential during optimization to avoid redundant API calls
- `dspy.inspect_history(n=5)` to inspect what prompts the optimizer is generating
- MIPROv2 with `auto="medium"` is a good default when you have enough data and care about quality
- Compiled programs store optimized few-shot demos and/or instructions — not model weights (unless using `BootstrapFinetune`)
