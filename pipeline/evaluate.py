"""
pipeline/evaluate.py — evaluate ClusterIntruderValidator with MLflow tracking.

Run:
    python pipeline/evaluate.py

Results are printed to stdout, saved to pipeline/eval_results.json, and
logged to the MLflow experiment "cluster-validator-eval".
"""

import json
from pathlib import Path

import dspy
import mlflow

from cluster_validator import (
    ClusterIntruderValidator,
    build_devset,
    configure_dspy,
    intruder_exact_match,
)

RESULTS_PATH = Path(__file__).parent.parent / "outputs" / "eval_results.json"
EXPERIMENT_NAME = "cluster-validator-eval"


def run_evaluation(config_path: str = "config/dspy_config.yaml", num_threads: int = 4) -> float:
    """Evaluate ClusterIntruderValidator and return accuracy (0–100).

    Logs traces, metrics, and results to MLflow automatically.
    """
    configure_dspy(config_path, cache=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.dspy.autolog(log_traces=True, log_traces_from_eval=True)

    devset = build_devset()
    program = ClusterIntruderValidator()

    with mlflow.start_run(run_name="evaluate"):
        evaluate = dspy.Evaluate(
            devset=devset,
            metric=intruder_exact_match,
            num_threads=num_threads,
            display_progress=True,
            display_table=10,
        )
        score = evaluate(program).score
        n_correct = int(round(score * len(devset) / 100))
        print(f"\nAccuracy: {score:.1f}%  ({n_correct}/{len(devset)} correct)")

        mlflow.log_metrics({"accuracy_pct": score, "n_correct": n_correct, "n_total": len(devset)})

        results = {
            "model": "see config/dspy_config.yaml",
            "num_examples": len(devset),
            "accuracy_pct": score,
        }
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        mlflow.log_artifact(str(RESULTS_PATH))
        print(f"Results saved to {RESULTS_PATH}")

        # Show first 5 failures for debugging
        print("\n--- Failure analysis (first 5 wrong) ---")
        n_shown = 0
        for ex in devset:
            if n_shown >= 5:
                break
            pred = program(**{k: getattr(ex, k) for k in
                              ("trefwoord_1", "trefwoord_2", "trefwoord_3", "trefwoord_4", "trefwoord_5", "trefwoord_6")})
            if not intruder_exact_match(ex, pred):
                keywords = [ex.trefwoord_1, ex.trefwoord_2, ex.trefwoord_3,
                            ex.trefwoord_4, ex.trefwoord_5, ex.trefwoord_6]
                print(f"  keywords : {keywords}")
                print(f"  gold     : {ex.indringer}")
                print(f"  pred     : {pred.indringer}")
                print()
                n_shown += 1
        if n_shown == 0:
            print("  No failures found — all examples answered correctly.")

    return score


if __name__ == "__main__":
    run_evaluation()
