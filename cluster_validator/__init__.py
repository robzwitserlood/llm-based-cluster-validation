"""cluster_validator — DSPy intruder-detection module and utilities."""

from .config import configure_dspy, configure_finetune_student_lm, configure_student_lm, configure_teacher_lm, get_finetuned_output_dir, load_config
from .data import build_devset, split_test, split_for_bootstrap, split_for_gepa, split_for_finetune
from .metrics import gepa_metric, intruder_exact_match
from .module import ClusterIntruderValidator, IntruderDetectionSignature, find_intruder

__all__ = [
    "ClusterIntruderValidator",
    "IntruderDetectionSignature",
    "find_intruder",
    "configure_dspy",
    "configure_finetune_student_lm",
    "configure_student_lm",
    "configure_teacher_lm",
    "get_finetuned_output_dir",
    "load_config",
    "build_devset",
    "split_test",
    "split_for_bootstrap",
    "split_for_gepa",
    "split_for_finetune",
    "intruder_exact_match",
    "gepa_metric",
]
