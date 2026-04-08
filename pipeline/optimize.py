"""
pipeline/optimize.py — optimize ClusterIntruderValidator with BootstrapFewShot.

Run:
    python pipeline/optimize.py

Output: outputs/program_optimized.json (and MLflow run in "cluster-validator-optimize")
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
    split_for_bootstrap,
    split_test,
)

OPTIMIZED_PATH = Path(__file__).parent.parent / "outputs" / "program_optimized.json"
EXPERIMENT_NAME = "cluster-validator-optimize"


def run_optimization(
    config_path: str = "config/dspy_config.yaml",
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 16,
    num_threads: int = 4,
) -> float:
    """Run BootstrapFewShot optimization and return test-set accuracy (0–100).

    Logs to MLflow experiment "cluster-validator-optimize".
    """
    configure_dspy(config_path, cache=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.dspy.autolog(log_traces=True, log_compiles=True)

    all_examples = build_devset()
    if len(all_examples) < 5:
        raise ValueError(
            f"Need at least 5 examples to optimize, got {len(all_examples)}. "
            "Run `python pipeline/build_dataset.py` to generate more."
        )

    trainable, testset = split_test(all_examples)
    trainset, devset = split_for_bootstrap(trainable)
    print(f"Split: {len(trainset)} train / {len(devset)} dev / {len(testset)} test")

    with mlflow.start_run(run_name="bootstrap-fewshot"):
        mlflow.log_params({
            "optimizer": "BootstrapFewShot",
            "num_train": len(trainset),
            "num_dev": len(devset),
            "num_test": len(testset),
            "max_bootstrapped_demos": max_bootstrapped_demos,
            "max_labeled_demos": max_labeled_demos,
        })

        evaluate_dev = dspy.Evaluate(
            devset=devset,
            metric=intruder_exact_match,
            num_threads=num_threads,
            display_progress=True,
            display_table=5,
        )
        evaluate_test = dspy.Evaluate(
            devset=testset,
            metric=intruder_exact_match,
            num_threads=num_threads,
            display_progress=True,
            display_table=0,
        )

        program = ClusterIntruderValidator()
        baseline_dev = evaluate_dev(program).score
        print(f"\nBaseline dev accuracy  : {baseline_dev:.1f}%")
        mlflow.log_metric("baseline_dev_accuracy_pct", baseline_dev)

        optimizer = dspy.BootstrapFewShot(
            metric=intruder_exact_match,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
        )
        print("\nRunning BootstrapFewShot …")
        compiled = optimizer.compile(program, trainset=trainset)

        optimized_dev = evaluate_dev(compiled).score
        optimized_test = evaluate_test(compiled).score
        print(f"\nOptimized dev accuracy : {optimized_dev:.1f}%  (delta: {optimized_dev - baseline_dev:+.1f}%)")
        print(f"Optimized test accuracy: {optimized_test:.1f}%")
        mlflow.log_metrics({
            "optimized_dev_accuracy_pct": optimized_dev,
            "optimized_test_accuracy_pct": optimized_test,
            "delta_dev_pct": optimized_dev - baseline_dev,
        })

        mlflow.dspy.log_model(
            compiled,
            artifact_path="dspy_program",
            task="llm/v1/chat",
            pip_requirements=["dspy"],
        )
        compiled.save(str(OPTIMIZED_PATH))
        mlflow.log_artifact(str(OPTIMIZED_PATH))
        print(f"\nCompiled program saved to {OPTIMIZED_PATH}")

        results_path = Path(__file__).parent.parent / "outputs" / "optimize_results.json"
        results_path.write_text(json.dumps({
            "optimizer": "BootstrapFewShot",
            "num_train": len(trainset),
            "num_dev": len(devset),
            "num_test": len(testset),
            "baseline_dev_accuracy_pct": baseline_dev,
            "optimized_dev_accuracy_pct": optimized_dev,
            "optimized_test_accuracy_pct": optimized_test,
        }, indent=2))
        print(f"Summary saved to {results_path}")

    return optimized_test


if __name__ == "__main__":
    run_optimization()
