"""
cluster_validator/config.py — DSPy LM configuration from dspy_config.yaml.
"""

import os
from pathlib import Path

import yaml
import dspy

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "dspy_config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _build_lm(model_cfg: dict, cache: bool = False) -> dspy.LM:
    """Construct a dspy.LM from a config block dict."""
    provider = model_cfg.get("provider", "openai")
    model_name = model_cfg["name"]
    temperature = model_cfg.get("temperature", 0.0)
    max_tokens = model_cfg.get("max_tokens", 1000)
    base_url = model_cfg.get("base_url")

    api_key_env = model_cfg.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else model_cfg.get("api_key")

    full_model_name = f"{provider}/{model_name}"

    lm_kwargs = dict(
        model=full_model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        cache=cache,
    )
    if api_key:
        lm_kwargs["api_key"] = api_key
    if base_url:
        lm_kwargs["base_url"] = base_url

    # Forward any extra sampling parameters (e.g. repetition_penalty)
    _KNOWN_KEYS = {"provider", "name", "temperature", "max_tokens", "base_url",
                   "api_key", "api_key_env"}
    for key, val in model_cfg.items():
        if key not in _KNOWN_KEYS:
            lm_kwargs[key] = val

    return dspy.LM(**lm_kwargs)


def configure_dspy(config_path: Path | str = DEFAULT_CONFIG_PATH, cache: bool = False) -> None:
    """Configure the global DSPy LM from the student block in dspy_config.yaml.

    Falls back to the legacy ``model`` key for backwards compatibility.

    Args:
        config_path: Path to the YAML configuration file.
        cache:       Enable DSPy response caching (useful for eval/optimize runs).
    """
    config = load_config(config_path)
    # Support both new "student" key and legacy "model" key
    model_cfg = config.get("student") or config["model"]

    lm = _build_lm(model_cfg, cache=cache)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

    full_model_name = f"{model_cfg.get('provider', 'openai')}/{model_cfg['name']}"
    print(f"[dspy] Configured LM: {full_model_name} (cache={cache})")


def configure_teacher_lm(config_path: Path | str = DEFAULT_CONFIG_PATH, cache: bool = False) -> dspy.LM:
    """Return a dspy.LM for the teacher model defined in dspy_config.yaml.

    Does NOT set it as the global DSPy LM — the caller decides where to attach it.

    Args:
        config_path: Path to the YAML configuration file.
        cache:       Enable DSPy response caching.

    Returns:
        A configured dspy.LM instance for the teacher.
    """
    config = load_config(config_path)
    if "teacher" not in config:
        raise KeyError(
            "No 'teacher' block found in dspy_config.yaml. "
            "Add a 'teacher:' section with provider, name, and api_key_env."
        )
    teacher_cfg = config["teacher"]
    lm = _build_lm(teacher_cfg, cache=cache)

    full_model_name = f"{teacher_cfg.get('provider', 'anthropic')}/{teacher_cfg['name']}"
    print(f"[dspy] Teacher LM    : {full_model_name} (cache={cache})")
    return lm
