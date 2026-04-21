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


_KNOWN_KEYS = {"provider", "name", "hf_name", "use_local_provider", "temperature",
               "max_tokens", "base_url", "api_key", "api_key_env"}


def _build_lm(model_cfg: dict, cache: bool = False) -> dspy.LM:
    """Construct a dspy.LM from a config block dict."""
    provider_str = model_cfg.get("provider", "openai")
    model_name = model_cfg["name"]
    hf_name = model_cfg.get("hf_name")
    use_local_provider = model_cfg.get("use_local_provider", False)
    temperature = model_cfg.get("temperature", 0.0)
    max_tokens = model_cfg.get("max_tokens", 1000)
    base_url = model_cfg.get("base_url")

    api_key_env = model_cfg.get("api_key_env")
    api_key = os.getenv(api_key_env) if api_key_env else model_cfg.get("api_key")

    if use_local_provider:
        # Use hf_name as the model path so LocalProvider.launch() passes the local
        # snapshot path to SGLang rather than downloading from HuggingFace.
        from dspy.clients.lm_local import LocalProvider
        model_path = hf_name or model_name
        full_model_name = f"openai/local:{model_path}"
        lm_kwargs = dict(
            model=full_model_name,
            finetuning_model=hf_name or model_name,
            provider=LocalProvider(),
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
        )
    else:
        full_model_name = f"{provider_str}/{model_name}"
        lm_kwargs = dict(
            model=full_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
        )

    if api_key:
        lm_kwargs["api_key"] = api_key
    if base_url and not use_local_provider:
        lm_kwargs["base_url"] = base_url

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


def configure_student_lm(config_path: Path | str = DEFAULT_CONFIG_PATH, cache: bool = False) -> dspy.LM:
    """Return a dspy.LM for the student model, with LocalProvider if use_local_provider is set.

    Does NOT set it as the global DSPy LM.
    """
    config = load_config(config_path)
    model_cfg = config.get("student") or config["model"]
    lm = _build_lm(model_cfg, cache=cache)

    model_label = model_cfg.get("name", model_cfg.get("hf_name", "unknown"))
    print(f"[dspy] Student LM    : {model_label} (cache={cache}, local_provider={model_cfg.get('use_local_provider', False)})")
    return lm


def get_finetuned_output_dir(config_path: Path | str = DEFAULT_CONFIG_PATH) -> str:
    """Return the output directory for fine-tuned weights from dspy_config.yaml."""
    config = load_config(config_path)
    return config.get("finetuned", {}).get("output_dir", "./finetuned_model")


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
