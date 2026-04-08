---
name: dspy-build
description: Scaffold a new DSPy Signature and Module for a described task. Use when the user wants to create a new DSPy program, add a new module, define a Signature, or implement a new AI capability with DSPy (Predict, ChainOfThought, ReAct).
argument-hint: [task description]
allowed-tools: Read Grep Glob
---

You are scaffolding a new DSPy Signature and Module. The user described the task as: `$ARGUMENTS`

## Step 1 — Read the existing codebase first

1. Glob `**/*.py` and read every file containing `import dspy` or `from dspy`
2. Read `dspy_config.yaml` (or any `*dspy*.yaml`) if present
3. Note existing Signature field names, Module class names, and predictor types — do not duplicate them

## Step 2 — Design the Signature

- Every field must have an explicit Python type and a `desc=` string
- Use `Literal["a", "b", ...]` for fixed-label classification outputs — never bare `str`
- Use `list[str]` or a Pydantic model / `TypedDict` for structured extraction outputs
- Input fields use `dspy.InputField(desc=...)`, output fields use `dspy.OutputField(desc=...)`

## Step 3 — Choose the right predictor

| Situation | Predictor |
|-----------|-----------|
| Simple lookup, no reasoning needed | `dspy.Predict` |
| Most tasks — adds automatic `reasoning` field | `dspy.ChainOfThought` (default) |
| Requires tool calls or multi-step decisions | `dspy.ReAct(signature, tools=[...])` |
| Self-consistency over N candidates | `dspy.MultiChainComparison` |

## Step 4 — Write the Module

Rules:
- Must call `super().__init__()`
- All predictors must be assigned as `self.` attributes — never instantiate inside `forward()`
- `forward()` accepts the same field names as the Signature's InputFields
- Return the predictor's result directly from `forward()`

## Step 5 — Output

Produce:
1. The `dspy.Signature` subclass
2. The `dspy.Module` subclass
3. A short usage snippet: `module = MyModule(); result = module(field=...)`
4. If `dspy.configure(lm=...)` is not already in the project, add a setup call

## DSPy idioms

- Never hardcode prompts as f-strings — Signatures manage prompt construction
- Set `temperature=0.0` for classification/extraction, `temperature=0.7`+ for generation
- Use `dspy.Assert(condition, msg)` / `dspy.Suggest(condition, msg)` to constrain outputs with automatic retry
- `dspy.inspect_history(n=5)` to debug the last N LM calls
- `dspy.LM(..., cache=True)` avoids re-running identical calls during development
