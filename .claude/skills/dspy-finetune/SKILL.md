---
name: dspy-finetune
description: Finetune LM weights with BootstrapFinetune. Use when the user wants to finetune a student model, set up a student/teacher pair, configure train_kwargs (LoRA/PEFT, epochs, learning rate), or run BootstrapFinetune.compile().
argument-hint: [module name or student model]
allowed-tools: Read Grep Glob
---

You are setting up a DSPy finetuning run with `BootstrapFinetune`. Target: `$ARGUMENTS` (if blank, find the main Module and use the configured LM as teacher).

## Step 1 — Read the existing codebase

1. Glob `**/*.py` and read every file containing `import dspy` or `from dspy`
2. Identify the Module to finetune and its Signature's InputField / OutputField names
3. Check for `dspy_config.yaml` — this is the source of truth for the teacher LM endpoint and model name
4. Look for existing metric functions and any saved `.json` compiled program files (load them into the teacher)
5. Check `outputs/` for a previously optimized program to use as the teacher

## Step 2 — Enable experimental mode

`BootstrapFinetune` is still experimental. This line must come before any compile:

```python
import dspy
dspy.settings.experimental = True
```

## Step 3 — Set up student and teacher programs

The student is the small model whose weights will be updated. The teacher is a capable model that generates training traces.

```python
from copy import deepcopy

# Teacher: the currently configured LM (or a stronger one like GPT-4o-mini)
# Load a previously compiled/optimized program if one exists — it produces better traces
teacher = MyModule()
teacher.load("outputs/program_optimized.json")  # omit if no compiled program exists

# Student: a smaller local model that will be finetuned
student = deepcopy(teacher)
student_lm = dspy.LM(
    model="openai/meta-llama/Llama-3.2-1B-Instruct",  # adjust to actual student model
    api_base="http://localhost:30000/v1",
    api_key="local",
    cache=False,  # do not cache — weights will change
)
student.set_lm(student_lm)
```

Key points:
- `deepcopy()` ensures student and teacher don't share state
- Set `cache=False` on the student LM — the point is to learn new weights, not replay cached calls
- The teacher LM should have `cache=True` to avoid redundant API calls during trace generation

## Step 4 — Pass a metric (this matters enormously)

Passing a metric to `BootstrapFinetune` filters training traces to only those the teacher got right. The DSPy tutorial shows: **51.5% accuracy without a metric → 86.7% with a metric** on the same training data.

Always pass `metric=` when labels are available:

```python
def metric(example, pred, trace=None):
    return pred.intruder.strip().lower() == example.intruder.strip().lower()

# Or import from cluster_validator if it already exists:
from cluster_validator.metrics import intruder_exact_match as metric
```

## Step 5 — Configure train_kwargs

```python
import os

# Set env vars before compile
os.environ["DSPY_FINETUNEDIR"] = "./outputs/finetune_checkpoints"  # separate from DSPy cache
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # restrict to one GPU; adjust for multi-GPU

train_kwargs = {
    "device": "cuda",
    "use_peft": True,           # LoRA — practical default on consumer GPUs
                                # set False only if you have >40 GB VRAM for full finetune
    "num_train_epochs": 3,
    "batch_size": 8,
    "learning_rate": 3e-4,      # standard LoRA starting point; lower (1e-4) for larger models
    "max_seq_length": 1024,     # keep short — intruder detection inputs are short
    "output_dir": "./outputs/finetuned_weights",
}

os.makedirs(train_kwargs["output_dir"], exist_ok=True)  # compile will fail if dir is missing
```

## Step 6 — Run compile

```python
optimizer = dspy.BootstrapFinetune(
    metric=metric,
    num_threads=16,
    train_kwargs=train_kwargs,
)

finetuned = optimizer.compile(
    student,
    teacher=teacher,
    trainset=trainset,   # list of dspy.Example with .with_inputs() set
)
```

`compile()` does three things in order:
1. Runs the teacher on all `trainset` inputs to generate labeled traces
2. Filters traces by `metric` (if provided)
3. Finetunes the student LM on the filtered traces

## Step 7 — Launch, evaluate, and compare

The finetuned program holds a new LM instance that must be explicitly launched before inference:

```python
try:
    finetuned.get_lm().launch()

    evaluate = dspy.Evaluate(
        devset=devset,
        metric=metric,
        num_threads=4,
        display_progress=True,
        display_table=5,
    )

    baseline_score = evaluate(teacher)
    finetuned_score = evaluate(finetuned)

    print(f"Teacher baseline : {baseline_score:.1f}%")
    print(f"Finetuned student: {finetuned_score:.1f}%")

finally:
    finetuned.get_lm().kill()   # always release the SGLang process
```

Always wrap `launch()` / `kill()` in `try/finally` — an uncaught exception will otherwise leave a zombie SGLang process.

## Step 8 — Save with MLflow

Prefer `mlflow.dspy.log_model()` over `program.save()` — it captures the full environment, pip dependencies, and a reusable artifact URI:

```python
import mlflow

mlflow.dspy.autolog(log_compiles=True)
mlflow.set_experiment("dspy-finetune")

with mlflow.start_run():
    mlflow.log_metric("teacher_baseline", baseline_score)
    mlflow.log_metric("finetuned_score", finetuned_score)
    mlflow.dspy.log_model(
        finetuned,
        artifact_path="dspy_program",
        task="llm/v1/chat",
        pip_requirements=["dspy"],
    )
```

Reload later:

```python
finetuned = mlflow.dspy.load_model("runs:/<run_id>/dspy_program")
finetuned.get_lm().launch()
result = finetuned(keywords=["fiets", "auto", "trein", "appel", "bus", "scooter"])
```

## Full script template

```python
import os
from copy import deepcopy
import dspy
import mlflow

dspy.settings.experimental = True

os.environ["DSPY_FINETUNEDIR"] = "./outputs/finetune_checkpoints"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# --- Configure LMs ---
from cluster_validator.config import configure_dspy
configure_dspy()  # sets teacher LM from dspy_config.yaml

from cluster_validator.module import ClusterIntruderValidator
from cluster_validator.metrics import intruder_exact_match as metric
from cluster_validator.data import build_devset

# --- Data ---
trainset = build_devset()  # or load a larger trainset
devset   = build_devset()  # ideally a separate held-out split

# --- Programs ---
teacher = ClusterIntruderValidator()
# teacher.load("outputs/program_optimized.json")  # uncomment if compiled program exists

student = deepcopy(teacher)
student_lm = dspy.LM(
    model="openai/ministral-4b-instruct",
    api_base="http://localhost:30000/v1",
    api_key="local",
    cache=False,
)
student.set_lm(student_lm)

# --- train_kwargs ---
train_kwargs = {
    "device": "cuda",
    "use_peft": True,
    "num_train_epochs": 3,
    "batch_size": 8,
    "learning_rate": 3e-4,
    "max_seq_length": 1024,
    "output_dir": "./outputs/finetuned_weights",
}
os.makedirs(train_kwargs["output_dir"], exist_ok=True)

# --- Compile ---
mlflow.dspy.autolog(log_compiles=True)
mlflow.set_experiment("dspy-finetune")

with mlflow.start_run():
    optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=16, train_kwargs=train_kwargs)
    finetuned = optimizer.compile(student, teacher=teacher, trainset=trainset)

    evaluate = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)

    try:
        finetuned.get_lm().launch()
        baseline_score = evaluate(teacher)
        finetuned_score = evaluate(finetuned)
        print(f"Teacher baseline : {baseline_score:.1f}%")
        print(f"Finetuned student: {finetuned_score:.1f}%")
    finally:
        finetuned.get_lm().kill()

    mlflow.log_metric("teacher_baseline", baseline_score)
    mlflow.log_metric("finetuned_score", finetuned_score)
    mlflow.dspy.log_model(finetuned, artifact_path="dspy_program", task="llm/v1/chat", pip_requirements=["dspy"])
```

## Common mistakes

| Mistake | Fix |
|--------|-----|
| Forgot `dspy.settings.experimental = True` | Add it before any import of `BootstrapFinetune` |
| `output_dir` does not exist | `os.makedirs(output_dir, exist_ok=True)` before compile |
| Student and teacher share the same LM object | Use `deepcopy(teacher)` then `student.set_lm(...)` |
| No metric passed | Always pass `metric=` when labels exist — impact is huge |
| SGLang process not killed on error | Wrap `launch()` + evaluation in `try/finally: get_lm().kill()` |
| `cache=True` on student LM | Student must have `cache=False` — you want fresh inference from updated weights |
