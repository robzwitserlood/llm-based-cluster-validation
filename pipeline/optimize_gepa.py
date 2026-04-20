"""
pipeline/optimize_gepa.py — optimize ClusterIntruderValidator with GEPA.

GEPA performs reflective prompt optimization: it runs the program, analyses
failures via a reflection LM, and iteratively proposes better instructions.

The reflection LM is read from the 'teacher' block in config/dspy_config.yaml
(currently claude-sonnet-4-6 via the Anthropic API). Requires ANTHROPIC_API_KEY.

Run:
    python pipeline/optimize_gepa.py

Output: outputs/program_gepa.json (and MLflow run in "cluster-validator-optimize")
"""

import argparse
import json
from pathlib import Path

import dspy
import mlflow

from cluster_validator import (
    ClusterIntruderValidator,
    build_devset,
    configure_dspy,
    configure_teacher_lm,
    gepa_metric,
    intruder_exact_match,
    split_for_gepa,
    split_test,
)

OPTIMIZED_PATH = Path(__file__).parent.parent / "outputs" / "program_gepa.json"
EXPERIMENT_NAME = "cluster-validator-optimize"


def run_optimization(
    auto: str = "light",
    config_path: str = "config/dspy_config.yaml",
    num_threads: int = 4,
    max_iterations: int | None = 3,
) -> float:
    """Run GEPA optimization and return test-set accuracy (0–100).

    Logs to MLflow experiment "cluster-validator-optimize".

    The reflection LM is read from the 'teacher' block in dspy_config.yaml.

    Args:
        auto:        GEPA budget preset: "light", "medium", or "heavy".
        config_path: Path to dspy_config.yaml for both the task and reflection LMs.
        num_threads: Parallel threads for evaluation inside GEPA.
    """
    configure_dspy(config_path, cache=True)
    reflection_lm = configure_teacher_lm(config_path, cache=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.dspy.autolog(log_traces=True, log_compiles=True)

    print(f"[gepa] Budget : auto={auto}")

    all_examples = build_devset()
    if len(all_examples) < 6:
        raise ValueError(
            f"Need at least 6 examples, got {len(all_examples)}. "
            "Run `python pipeline/build_dataset.py` to generate more."
        )

    trainable, testset = split_test(all_examples)
    trainset, valset, devset = split_for_gepa(trainable)
    print(f"Split: {len(trainset)} train / {len(valset)} val / {len(devset)} dev / {len(testset)} test")

    with mlflow.start_run(run_name="gepa"):
        mlflow.log_params({
            "optimizer": "GEPA",
            "reflection_lm": "claude-sonnet-4-6",
            "gepa_auto": auto,
            "gepa_max_iterations": max_iterations,
            "num_train": len(trainset),
            "num_val": len(valset),
            "num_dev": len(devset),
            "num_test": len(testset),
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

        optimizer = dspy.GEPA(
            metric=gepa_metric,
            auto=auto,
            max_full_evals=max_iterations,
            reflection_lm=reflection_lm,
            num_threads=num_threads,
        )
        print("\nRunning GEPA …")
        compiled = optimizer.compile(
            student=ClusterIntruderValidator(),
            trainset=trainset,
            valset=valset,
        )

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

        results_path = Path(__file__).parent.parent / "outputs" / "optimize_gepa_results.json"
        results_path.write_text(json.dumps({
            "optimizer": "GEPA",
            "reflection_lm": "claude-sonnet-4-6",
            "gepa_auto": auto,
            "gepa_max_iterations": max_iterations,
            "num_train": len(trainset),
            "num_val": len(valset),
            "num_dev": len(devset),
            "num_test": len(testset),
            "baseline_dev_accuracy_pct": baseline_dev,
            "optimized_dev_accuracy_pct": optimized_dev,
            "optimized_test_accuracy_pct": optimized_test,
        }, indent=2))
        print(f"Summary saved to {results_path}")

    return optimized_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    p.add_argument("--config-path", default="config/dspy_config.yaml")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--max-iterations", type=int, default=3,
                   help="Max GEPA full-eval iterations (None = use auto budget only).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_optimization(
        auto=args.auto,
        config_path=args.config_path,
        num_threads=args.num_threads,
        max_iterations=args.max_iterations,
    )
