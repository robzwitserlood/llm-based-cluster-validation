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


def configure_dspy(config_path: Path | str = DEFAULT_CONFIG_PATH, cache: bool = False) -> None:
    """Configure the global DSPy LM from dspy_config.yaml.

    Args:
        config_path: Path to the YAML configuration file.
        cache:       Enable DSPy response caching (useful for eval/optimize runs).
    """
    config = load_config(config_path)
    model_cfg = config["model"]

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

    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)
    print(f"[dspy] Configured LM: {full_model_name} (cache={cache})")
