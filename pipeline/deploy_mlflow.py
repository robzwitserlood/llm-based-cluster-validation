"""
pipeline/deploy_mlflow.py — log the fine-tuned ClusterIntruderValidator to MLflow.

Prerequisites:
  - Run pipeline/optimize_finetune.py first to produce:
      ./finetuned_model/                      (HF weights)
      pipeline/program_finetune.json          (compiled DSPy program)
      pipeline/optimize_finetune_results.json

Run:
    python pipeline/deploy_mlflow.py
    python pipeline/deploy_mlflow.py --output-dir ./finetuned_model --mlflow-uri http://localhost:5000

Serve (after logging):
    mlflow models serve -m "runs:/<run_id>/dspy_program" -p 6000
"""

import argparse
import json
import pathlib

import dspy
import mlflow
from dspy.clients.lm_local import LocalProvider

from cluster_validator import ClusterIntruderValidator

OUTPUTS_DIR = pathlib.Path(__file__).parent.parent / "outputs"
FINETUNED_PROGRAM_PATH = OUTPUTS_DIR / "program_finetune.json"
FINETUNE_RESULTS_PATH = OUTPUTS_DIR / "optimize_finetune_results.json"
EXPERIMENT_NAME = "cluster-validator-finetuned"


# ---------------------------------------------------------------------------
# MLflow-compatible Module wrapper
# MLflow 2.22.0+ requires a dspy.Module subclass with forward(self, messages)
# when using task="llm/v1/chat". Input: comma-separated keyword string.
# ---------------------------------------------------------------------------

class FinetunedClusterValidator(dspy.Module):
    """Wraps ClusterIntruderValidator to fit MLflow's llm/v1/chat schema."""

    def __init__(self, student_lm: dspy.LM, program_path: pathlib.Path = FINETUNED_PROGRAM_PATH) -> None:
        super().__init__()
        self.program = ClusterIntruderValidator()
        if program_path.exists():
            self.program.load(str(program_path))
            print(f"[deploy] Loaded compiled program from {program_path}")
        else:
            print(f"[deploy] WARNING: {program_path} not found — using uncompiled program")
        self.program.predictor.set_lm(student_lm)

    def forward(self, messages: list[dict]) -> dict:
        """
        Args:
            messages: OpenAI-style chat messages list.
                      The first user message must contain six comma-separated keywords,
                      e.g. "ocean,river,lake,mountain,stream,pond"
        Returns:
            {"intruder": "<keyword>"}
        """
        content = messages[0]["content"]
        keywords = [kw.strip() for kw in content.split(",")]
        if len(keywords) != 6:
            raise ValueError(
                f"Expected 6 comma-separated keywords, got {len(keywords)}: {content!r}"
            )
        result = self.program(
            trefwoord_1=keywords[0], trefwoord_2=keywords[1], trefwoord_3=keywords[2],
            trefwoord_4=keywords[3], trefwoord_5=keywords[4], trefwoord_6=keywords[5],
        )
        return {"indringer": result.indringer.strip()}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log_to_mlflow(output_dir: str = "./finetuned_model") -> str:
    """Log the fine-tuned program to MLflow. Returns the MLflow run ID."""
    student_lm = dspy.LM(
        model=f"openai/local:{output_dir}",
        provider=LocalProvider(),
        max_tokens=1000,
        temperature=0.0,
        cache=True,
    )

    print("[deploy] Launching student inference server (SGLang) …")
    student_lm.launch()

    try:
        program = FinetunedClusterValidator(student_lm=student_lm)

        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="ministral-4b-finetuned") as run:
            if FINETUNE_RESULTS_PATH.exists():
                results = json.loads(FINETUNE_RESULTS_PATH.read_text())
                mlflow.log_metrics({
                    "teacher_baseline_pct": results["teacher_baseline_dev_pct"],
                    "student_pre_finetune_pct": results["student_pre_finetune_dev_pct"],
                    "student_post_finetune_pct": results["student_post_finetune_dev_pct"],
                    "delta_vs_teacher_pct": results["delta_vs_teacher_dev_pct"],
                })
                mlflow.log_params({
                    "teacher_lm": results.get("teacher_model", "see config/dspy_config.yaml"),
                    "student_model": results.get("student_model", output_dir),
                    "num_train": results["num_train"],
                    "num_dev": results["num_dev"],
                })
                print(f"[deploy] Logged metrics from {FINETUNE_RESULTS_PATH}")

            if FINETUNED_PROGRAM_PATH.exists():
                mlflow.log_artifact(str(FINETUNED_PROGRAM_PATH))

            mlflow.dspy.log_model(
                program,
                artifact_path="dspy_program",
                task="llm/v1/chat",
                input_example={
                    "messages": [{"role": "user", "content": "ocean,river,lake,mountain,stream,pond"}]
                },
                pip_requirements=[
                    "dspy", "mlflow>=2.18.0",
                    "sglang[all]>=0.4.4.post3", "transformers", "accelerate", "peft",
                ],
            )

            run_id = run.info.run_id
            print(f"\n[deploy] Run ID     : {run_id}")
            print(f"[deploy] Experiment : {EXPERIMENT_NAME}")
            print(f"\nTo serve:\n  mlflow models serve -m runs:/{run_id}/dspy_program -p 6000")

    finally:
        print("[deploy] Shutting down student inference server …")
        student_lm.kill()

    return run_id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir",  default="./finetuned_model")
    p.add_argument("--mlflow-uri",  default=None, help="MLflow tracking URI (default: local ./mlruns)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not pathlib.Path(args.output_dir).exists():
        raise FileNotFoundError(
            f"Fine-tuned model directory not found: {args.output_dir}\n"
            "Run pipeline/optimize_finetune.py first."
        )
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    log_to_mlflow(args.output_dir)
