# Snakefile — orchestrates the DSPy cluster-validation pipeline.
#
# Targets:
#   snakemake all              # build_dataset → evaluate, optimize, optimize_gepa → optimize_finetune → deploy
#   snakemake evaluate         # just the baseline evaluation
#   snakemake optimize         # BootstrapFewShot
#   snakemake optimize_gepa    # GEPA (needs ANTHROPIC_API_KEY)
#   snakemake optimize_finetune  # BootstrapFinetune (needs ANTHROPIC_API_KEY, kills SGLang)
#   snakemake deploy           # log fine-tuned program to MLflow
#   snakemake clean            # remove outputs/ and finetuned_model/
#
# Override any config value per run:
#   snakemake --config gepa_auto=heavy num_threads=8 optimize_gepa
#
# Inspect the DAG without running anything:
#   snakemake --dry-run all
#   snakemake --dag all | dot -Tpng > dag.png
import sys
from pathlib import Path

REPO_ROOT = Path(workflow.basedir).resolve()
# Make `pipeline/*` and `scripts/*` importable from run: blocks.
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

configfile: "config/pipeline_config.yaml"

DSPY_CONFIG      = config["dspy_config"]
NUM_THREADS      = config["num_threads"]
FINETUNE         = config["finetune"]
FINETUNED_DIR    = FINETUNE["output_dir"]

DATA_DIR         = "data"
OUTPUTS_DIR      = "outputs"
RAW_EXAMPLES     = f"{DATA_DIR}/raw_examples.json"
TOPICS           = f"{DATA_DIR}/topics.json"

EVAL_RESULTS     = f"{OUTPUTS_DIR}/eval_results.json"
OPT_PROGRAM      = f"{OUTPUTS_DIR}/program_optimized.json"
OPT_RESULTS      = f"{OUTPUTS_DIR}/optimize_results.json"
GEPA_PROGRAM     = f"{OUTPUTS_DIR}/program_gepa.json"
GEPA_RESULTS     = f"{OUTPUTS_DIR}/optimize_gepa_results.json"
FT_PROGRAM       = f"{OUTPUTS_DIR}/program_finetune.json"
FT_RESULTS       = f"{OUTPUTS_DIR}/optimize_finetune_results.json"
DEPLOY_MARKER    = f"{OUTPUTS_DIR}/.deploy_marker"


# ---------------------------------------------------------------------------
# End-to-end target — runs every stage.
# ---------------------------------------------------------------------------
rule all:
    input:
        EVAL_RESULTS,
        OPT_PROGRAM,
        GEPA_PROGRAM,
        FT_PROGRAM,
        DEPLOY_MARKER,


# ---------------------------------------------------------------------------
# Build the dataset (Dutch news → BERTopic → raw_examples.json).
# No GPU inference; BERTopic embedding runs locally on CPU/GPU either way.
# ---------------------------------------------------------------------------
rule build_dataset:
    output:
        examples=RAW_EXAMPLES,
        topics=TOPICS,
    params:
        max_docs=config["max_docs"],
        examples_per_topic=config["examples_per_topic"],
        top_n=config["top_n"],
        seed=config["seed"],
    run:
        from pipeline.build_dataset import run_build_dataset
        run_build_dataset(
            out=Path(output.topics),
            examples_out=Path(output.examples),
            max_docs=params.max_docs,
            examples_per_topic=params.examples_per_topic,
            top_n=params.top_n,
            seed=params.seed,
        )


# ---------------------------------------------------------------------------
# Baseline evaluation — needs SGLang up.
# ---------------------------------------------------------------------------
rule evaluate:
    input: RAW_EXAMPLES
    output: EVAL_RESULTS
    resources: gpu=1
    run:
        from sglang_manager import ensure_running
        from pipeline.evaluate import run_evaluation
        ensure_running()
        run_evaluation(config_path=DSPY_CONFIG, num_threads=NUM_THREADS)


# ---------------------------------------------------------------------------
# BootstrapFewShot — needs SGLang up.
# ---------------------------------------------------------------------------
rule optimize:
    input: RAW_EXAMPLES
    output:
        program=OPT_PROGRAM,
        results=OPT_RESULTS,
    resources: gpu=1
    params:
        max_bootstrapped_demos=config["max_bootstrapped_demos"],
        max_labeled_demos=config["max_labeled_demos"],
    run:
        from sglang_manager import ensure_running
        from pipeline.optimize import run_optimization
        ensure_running()
        run_optimization(
            config_path=DSPY_CONFIG,
            max_bootstrapped_demos=params.max_bootstrapped_demos,
            max_labeled_demos=params.max_labeled_demos,
            num_threads=NUM_THREADS,
        )


# ---------------------------------------------------------------------------
# GEPA — needs SGLang up + ANTHROPIC_API_KEY for the reflection LM.
# ---------------------------------------------------------------------------
rule optimize_gepa:
    input: RAW_EXAMPLES
    output:
        program=GEPA_PROGRAM,
        results=GEPA_RESULTS,
    resources: gpu=1
    params:
        auto=config["gepa_auto"],
        max_iterations=config["gepa_max_iterations"],
    run:
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError("ANTHROPIC_API_KEY is required for optimize_gepa")
        from sglang_manager import ensure_running
        from pipeline.optimize_gepa import run_optimization
        ensure_running()
        run_optimization(
            auto=params.auto,
            config_path=DSPY_CONFIG,
            num_threads=NUM_THREADS,
            max_iterations=params.max_iterations,
        )


# ---------------------------------------------------------------------------
# BootstrapFinetune — needs SGLang DOWN (manages its own server lifecycle for
# pre- and post-finetune evaluation) + ANTHROPIC_API_KEY for the teacher.
# ---------------------------------------------------------------------------
rule optimize_finetune:
    input: RAW_EXAMPLES
    output:
        weights=directory(FINETUNED_DIR),
        program=FT_PROGRAM,
        results=FT_RESULTS,
    resources: gpu=1
    params:
        output_dir=FINETUNED_DIR,
        epochs=FINETUNE["epochs"],
        lr=FINETUNE["lr"],
        batch_size=FINETUNE["batch_size"],
        grad_accum=FINETUNE["grad_accum"],
        bf16=FINETUNE["bf16"],
        use_peft=FINETUNE["use_peft"],
    run:
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise EnvironmentError("ANTHROPIC_API_KEY is required for optimize_finetune")
        from sglang_manager import kill
        from pipeline.optimize_finetune import run_optimization
        kill()
        run_optimization(
            output_dir=params.output_dir,
            config_path=DSPY_CONFIG,
            epochs=params.epochs,
            lr=params.lr,
            batch_size=params.batch_size,
            grad_accum=params.grad_accum,
            bf16=params.bf16,
            use_peft=params.use_peft,
            num_threads=NUM_THREADS,
        )


# ---------------------------------------------------------------------------
# Deploy to MLflow — deploy_mlflow launches its own SGLang server, so kill
# any existing one first to avoid a port collision.
# ---------------------------------------------------------------------------
rule deploy_mlflow:
    input:
        weights=FINETUNED_DIR,
        program=FT_PROGRAM,
    output: DEPLOY_MARKER
    resources: gpu=1
    run:
        from sglang_manager import kill
        from pipeline.deploy_mlflow import log_to_mlflow
        kill()
        run_id = log_to_mlflow(output_dir=input.weights)
        Path(output[0]).write_text(run_id + "\n")


# Friendly alias
rule deploy:
    input: DEPLOY_MARKER


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
rule clean:
    shell:
        "rm -rf outputs/ finetuned_model/"
