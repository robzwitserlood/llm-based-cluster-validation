"""
cluster_validator/config.py — DSPy LM configuration from dspy_config.yaml.
"""

import os
import time
import urllib.request
import urllib.error
from pathlib import Path

import yaml
import dspy
import trl
from dspy.clients.lm_local import LocalProvider

# trl>=0.30 removed `setup_chat_format`, but DSPy 3.1's train_sft_locally still
# imports it. The call site wraps it in `try/except Exception: pass`, so a no-op
# shim is sufficient to let the import succeed.
if not hasattr(trl, "setup_chat_format"):
    trl.setup_chat_format = lambda model=None, tokenizer=None, **_: (model, tokenizer)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "dspy_config.yaml"


class PatchedLocalProvider(LocalProvider):
    """LocalProvider that avoids the `local:` prefix in the returned model name.

    Upstream LocalProvider.finetune() returns `openai/local:<output_dir>` after training.
    The `local:` prefix survives into the fine-tuned LM's model field and is sent
    verbatim to SGLang, which (>=0.4) interprets the colon as `base-model:adapter-name`
    and raises "LoRA adapter '<output_dir>' was requested, but LoRA is not enabled".
    """

    @staticmethod
    def finetune(job, model, train_data, train_data_format, train_kwargs=None) -> str:
        result = LocalProvider.finetune(
            job=job,
            model=model,
            train_data=train_data,
            train_data_format=train_data_format,
            train_kwargs=train_kwargs,
        )
        if result.startswith("openai/local:"):
            result = "openai/" + result[len("openai/local:"):]
        return result


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


_KNOWN_KEYS = {"provider", "name", "hf_name", "use_local_provider", "temperature",
               "max_tokens", "base_url", "api_key", "api_key_env"}

# Threads to use when the LM is a local server (single GPU — no point saturating it)
LOCAL_NUM_THREADS = 2
# Threads for remote/cloud LMs
REMOTE_NUM_THREADS = 8


def wait_for_server(base_url: str, timeout: int = 120, interval: float = 3.0) -> None:
    """Poll base_url/health until the server responds 200 or timeout expires.

    Raises RuntimeError if the server is not reachable within timeout seconds.
    """
    health_url = base_url.rstrip("/").removesuffix("/v1") + "/health"
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=3) as resp:
                if resp.status == 200:
                    print(f"[dspy] Server ready at {health_url}")
                    return
        except Exception as exc:
            last_exc = exc
        time.sleep(interval)
    raise RuntimeError(
        f"Server at {health_url} did not become ready within {timeout}s. "
        f"Last error: {last_exc}"
    )


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
        # NOTE: we deliberately do NOT use DSPy's default "openai/local:<path>" form.
        # SGLang >=0.4 parses a colon in the OpenAI `model` field as
        # `base-model:adapter-name` for multi-LoRA serving, which turns the HF snapshot
        # path into a bogus LoRA adapter request and fails with "LoRA is not enabled".
        model_path = hf_name or model_name
        full_model_name = f"openai/{model_path}"
        lm_kwargs = dict(
            model=full_model_name,
            finetuning_model=hf_name or model_name,
            provider=PatchedLocalProvider(),
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            num_retries=3,
        )
    else:
        full_model_name = f"{provider_str}/{model_name}"
        lm_kwargs = dict(
            model=full_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            num_retries=3,
        )

    if api_key:
        lm_kwargs["api_key"] = api_key
    if base_url and not use_local_provider:
        lm_kwargs["base_url"] = base_url

    for key, val in model_cfg.items():
        if key not in _KNOWN_KEYS:
            lm_kwargs[key] = val

    return dspy.LM(**lm_kwargs)


def configure_dspy(
    config_path: Path | str = DEFAULT_CONFIG_PATH,
    cache: bool = False,
    wait_for_server_ready: bool = True,
    server_timeout: int = 120,
) -> int:
    """Configure the global DSPy LM from the student block in dspy_config.yaml.

    Falls back to the legacy ``model`` key for backwards compatibility.

    Args:
        config_path:          Path to the YAML configuration file.
        cache:                Enable DSPy response caching (useful for eval/optimize runs).
        wait_for_server_ready: Poll the server health endpoint until ready (local servers only).
        server_timeout:       Seconds to wait for the server before raising.

    Returns:
        Recommended num_threads for dspy.Evaluate — lower for local servers.
    """
    config = load_config(config_path)
    # Support both new "student" key and legacy "model" key
    model_cfg = config.get("student") or config["model"]

    is_local = bool(model_cfg.get("base_url") or model_cfg.get("use_local_provider"))
    if wait_for_server_ready and is_local:
        base_url = model_cfg.get("base_url", "http://localhost:30000/v1")
        wait_for_server(base_url, timeout=server_timeout)

    lm = _build_lm(model_cfg, cache=cache)
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

    full_model_name = f"{model_cfg.get('provider', 'openai')}/{model_cfg['name']}"
    num_threads = LOCAL_NUM_THREADS if is_local else REMOTE_NUM_THREADS
    print(f"[dspy] Configured LM: {full_model_name} (cache={cache}, num_threads={num_threads})")
    return num_threads


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


def configure_finetune_student_lm(config_path: Path | str = DEFAULT_CONFIG_PATH, cache: bool = False) -> dspy.LM:
    """Return a dspy.LM for the finetune_student block in dspy_config.yaml.

    This block must use SGLang with use_local_provider=true so that
    launch() / kill() / finetuning are available. It is intentionally
    separate from the 'student' block so that evaluate/optimize scripts
    can point 'student' at any OpenAI-compatible server (lmstudio, etc.)
    without affecting the finetune workflow.

    Does NOT set the global DSPy LM.
    """
    config = load_config(config_path)
    if "finetune_student" not in config:
        raise KeyError(
            "No 'finetune_student' block found in dspy_config.yaml. "
            "Add a 'finetune_student:' section with use_local_provider: true "
            "and hf_name pointing to the local model snapshot."
        )
    ft_cfg = config["finetune_student"]
    lm = _build_lm(ft_cfg, cache=cache)

    model_label = ft_cfg.get("name", ft_cfg.get("hf_name", "unknown"))
    print(f"[dspy] Finetune student LM: {model_label} (cache={cache}, local_provider={ft_cfg.get('use_local_provider', False)})")
    return lm


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
