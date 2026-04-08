"""
pipeline/optimize_finetune.py — fine-tune ClusterIntruderValidator with BootstrapFinetune.

BootstrapFinetune works in two phases:
  1. Bootstrap: run a *teacher* program on the trainset to collect successful
     (prompt, completion) traces.
  2. Finetune: use those traces to fine-tune the *student* model's weights via
     the HuggingFace TRL/PEFT stack.

The student LM is a local HuggingFace model served by SGLang; the teacher is
the model from config/dspy_config.yaml (ministral-3:3b via SGLang).

Run:
    python pipeline/optimize_finetune.py
    python pipeline/optimize_finetune.py \\
        --student-model meta-llama/Llama-3.2-1B-Instruct \\
        --output-dir ./finetuned_model

Output:
  - ./finetuned_model/                        (HF model weights)
  - outputs/program_finetune.json
  - outputs/optimize_finetune_results.json
  - MLflow run in "cluster-validator-finetune"
"""

import argparse
import json
from pathlib import Path

import dspy
import mlflow
from dspy.clients.lm_local import LocalProvider

from cluster_validator import (
    ClusterIntruderValidator,
    build_devset,
    configure_dspy,
    intruder_exact_match,
    split_for_finetune,
    split_test,
)

OPTIMIZED_PATH = Path(__file__).parent.parent / "outputs" / "program_finetune.json"
EXPERIMENT_NAME = "cluster-validator-finetune"


def run_optimization(
    student_model: str = "meta-llama/Llama-3.2-1B-Instruct",
    output_dir: str = "./finetuned_model",
    config_path: str = "config/dspy_config.yaml",
    epochs: int = 1,
    lr: float = 5e-5,
    batch_size: int = 4,
    grad_accum: int = 4,
    bf16: bool = False,
    use_peft: bool = False,
    num_threads: int = 4,
) -> float:
    """Run BootstrapFinetune and return test-set accuracy (0–100).

    Logs to MLflow experiment "cluster-validator-finetune".

    Args:
        student_model: HuggingFace model ID to fine-tune (the student).
        output_dir:    Directory where fine-tuned weights are saved.
        config_path:   Path to dspy_config.yaml for the teacher LM.
        epochs:        Fine-tuning epochs.
        lr:            Learning rate.
        batch_size:    Per-device train batch size.
        grad_accum:    Gradient accumulation steps.
        bf16:          Use bfloat16 mixed precision.
        use_peft:      Use PEFT/LoRA for memory-efficient fine-tuning.
        num_threads:   Threads for evaluation.
    """
    dspy.settings.experimental = True

    configure_dspy(config_path, cache=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.dspy.autolog(log_traces=True, log_compiles=True)

    teacher_program = ClusterIntruderValidator()

    train_kwargs = {
        "num_train_epochs": epochs,
        "learning_rate": lr,
        "per_device_train_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "output_dir": output_dir,
        "bf16": bf16,
        "use_peft": use_peft,
    }
    student_lm = dspy.LM(
        model=f"openai/local:{student_model}",
        provider=LocalProvider(),
        max_tokens=1000,
        temperature=0.0,
        cache=True,
    )
    student_program = ClusterIntruderValidator()
    student_program.predictor.set_lm(student_lm)

    print(f"[finetune] Teacher LM : see {config_path}")
    print(f"[finetune] Student LM : {student_model}")
    print(f"[finetune] Output dir : {output_dir}")

    all_examples = build_devset()
    if len(all_examples) < 5:
        raise ValueError(
            f"Need at least 5 examples, got {len(all_examples)}. "
            "Run `python pipeline/build_dataset.py` to generate more."
        )

    trainable, testset = split_test(all_examples)
    trainset, devset = split_for_finetune(trainable)
    print(f"Split: {len(trainset)} train / {len(devset)} dev / {len(testset)} test")

    with mlflow.start_run(run_name="bootstrap-finetune"):
        mlflow.log_params({
            "optimizer": "BootstrapFinetune",
            "student_model": student_model,
            "output_dir": output_dir,
            "epochs": epochs,
            "lr": lr,
            "batch_size": batch_size,
            "use_peft": use_peft,
            "num_train": len(trainset),
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

        baseline_dev = evaluate_dev(teacher_program).score
        print(f"\nBaseline dev accuracy (teacher) : {baseline_dev:.1f}%")
        mlflow.log_metric("teacher_baseline_dev_pct", baseline_dev)

        print("\nLaunching student inference server (SGLang) …")
        student_lm.launch()

        try:
            student_pre_dev = evaluate_dev(student_program).score
            print(f"Student pre-finetune dev        : {student_pre_dev:.1f}%")
            mlflow.log_metric("student_pre_finetune_dev_pct", student_pre_dev)

            optimizer = dspy.BootstrapFinetune(
                metric=intruder_exact_match,
                train_kwargs=train_kwargs,
                num_threads=num_threads,
            )
            print("\nRunning BootstrapFinetune …")
            compiled = optimizer.compile(
                student=student_program,
                trainset=trainset,
                teacher=teacher_program,
            )

            student_post_dev = evaluate_dev(compiled).score
            student_post_test = evaluate_test(compiled).score
            print(f"\nStudent post-finetune dev  : {student_post_dev:.1f}%  (delta vs pre: {student_post_dev - student_pre_dev:+.1f}%)")
            print(f"Student post-finetune test : {student_post_test:.1f}%")
            mlflow.log_metrics({
                "student_post_finetune_dev_pct": student_post_dev,
                "student_post_finetune_test_pct": student_post_test,
                "delta_vs_pre_dev_pct": student_post_dev - student_pre_dev,
                "delta_vs_teacher_dev_pct": student_post_dev - baseline_dev,
            })

            mlflow.dspy.log_model(
                compiled,
                artifact_path="dspy_program",
                task="llm/v1/chat",
                pip_requirements=["dspy", "sglang[all]>=0.4.4.post3", "transformers", "accelerate", "peft"],
            )
            mlflow.log_artifact(output_dir, artifact_path="finetuned_weights")
            compiled.save(str(OPTIMIZED_PATH))
            mlflow.log_artifact(str(OPTIMIZED_PATH))
            print(f"\nCompiled program saved to {OPTIMIZED_PATH}")

            results_path = Path(__file__).parent.parent / "outputs" / "optimize_finetune_results.json"
            results_path.write_text(json.dumps({
                "teacher_lm": f"see {config_path}",
                "student_model": student_model,
                "output_dir": output_dir,
                "num_train": len(trainset),
                "num_dev": len(devset),
                "num_test": len(testset),
                "teacher_baseline_dev_pct": baseline_dev,
                "student_pre_finetune_dev_pct": student_pre_dev,
                "student_post_finetune_dev_pct": student_post_dev,
                "student_post_finetune_test_pct": student_post_test,
                "delta_vs_pre_dev_pct": student_post_dev - student_pre_dev,
                "delta_vs_teacher_dev_pct": student_post_dev - baseline_dev,
            }, indent=2))
            print(f"Summary saved to {results_path}")

        finally:
            print("\nShutting down student inference server …")
            student_lm.kill()

    return student_post_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--student-model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--output-dir",    default="./finetuned_model")
    p.add_argument("--config-path",   default="config/dspy_config.yaml")
    p.add_argument("--epochs",        type=int,   default=1)
    p.add_argument("--lr",            type=float, default=5e-5)
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--grad-accum",    type=int,   default=4)
    p.add_argument("--bf16",          action="store_true")
    p.add_argument("--use-peft",      action="store_true")
    p.add_argument("--num-threads",   type=int,   default=4)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_optimization(
        student_model=args.student_model,
        output_dir=args.output_dir,
        config_path=args.config_path,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        bf16=args.bf16,
        use_peft=args.use_peft,
        num_threads=args.num_threads,
    )
