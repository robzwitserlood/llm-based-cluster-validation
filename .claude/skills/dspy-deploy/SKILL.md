---
name: dspy-deploy
description: Deploy a DSPy program to production. Use when the user wants to serve a DSPy module as an API endpoint, set up FastAPI with dspy.asyncify, use dspy.streamify for streaming, or package and version a program with MLflow.
allowed-tools: Read Grep Glob
---

You are deploying a DSPy program. Target: `$ARGUMENTS` (if blank, ask: FastAPI, MLflow, or streaming?).

## Step 1 — Read the existing codebase

1. Glob `**/*.py` and read every file containing `import dspy` or `from dspy`
2. Identify the Module to deploy and its Signature's InputField / OutputField names
3. Check for a saved `*.json` compiled program file to load
4. Read `dspy_config.yaml` if present for LM settings

## Step 2 — Dispatch on target

Use MLflow (`>=2.18.0`) for program versioning, environment management, and standardised LLM API compatibility.

**Requirement (MLflow 2.22.0+):** the program must be a `dspy.Module` subclass — bare `dspy.Predict` / `dspy.ChainOfThought` instances cannot be logged directly. Wrap them if needed:

```python
class MyProgram(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought("question -> answer")

    def forward(self, messages):
        # MLflow's llm/v1/chat task passes input as messages list
        return self.cot(question=messages[0]["content"])
```

Log and serve:

```python
import mlflow
import dspy

mlflow.set_experiment("my-dspy-app")

with mlflow.start_run():
    mlflow.dspy.log_model(
        program,
        artifact_path="dspy_program",
        task="llm/v1/chat",
        input_example={"messages": [{"role": "user", "content": "What is 2+2?"}]},
        pip_requirements=["dspy"],
    )
    # Optionally log evaluation metrics from the same run
    mlflow.log_metric("dev_accuracy", dev_score)
```

Start the MLflow UI and serve:
```bash
mlflow ui                                              # http://127.0.0.1:5000
mlflow models serve -m "runs:/<run_id>/dspy_program" -p 6000
```

## General deployment rules

- Always load a compiled (optimized) program file if one exists: `program.load("program_optimized.json")`
- Set `cache=True` in `dspy.LM(...)` to avoid redundant LM calls for repeated inputs
- Use Pydantic `BaseModel` for request/response validation — mirror field names exactly from the Signature
- Do not expose raw `dspy.Prediction` objects in API responses; extract and serialize the fields explicitly
- Log the DSPy program version and LM model name in the MLflow run or FastAPI startup event for traceability
