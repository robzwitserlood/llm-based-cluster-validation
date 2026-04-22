"""SGLang server lifecycle helpers for pipeline orchestration.

The Snakefile uses these to ensure SGLang is running before inference stages
(evaluate, optimize, optimize_gepa) and killed before stages that need the
full GPU (optimize_finetune, deploy_mlflow — those manage their own server).

Idempotent: ``ensure_running()`` is a no-op if the server is already healthy;
``kill()`` is a no-op if nothing is running.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HEALTH_URL = os.environ.get("SGLANG_HEALTH_URL", "http://localhost:30000/health")
DEFAULT_PORT = os.environ.get("SGLANG_PORT", "30000")
DEFAULT_MODEL_PATH = os.environ.get("SGLANG_MODEL_PATH", "ministral/Ministral-4b-instruct")
LAUNCH_TIMEOUT_S = int(os.environ.get("SGLANG_LAUNCH_TIMEOUT_S", "300"))
LOG_PATH = Path(os.environ.get("SGLANG_LOG_PATH", "outputs/sglang.log"))


def is_running(url: str = DEFAULT_HEALTH_URL, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False


def ensure_running(
    model_path: str = DEFAULT_MODEL_PATH,
    port: str = DEFAULT_PORT,
    health_url: str = DEFAULT_HEALTH_URL,
    timeout_s: int = LAUNCH_TIMEOUT_S,
) -> None:
    """Launch SGLang if it's not already serving /health on the given URL."""
    if is_running(health_url):
        print(f"[sglang] already running at {health_url}")
        return

    print(f"[sglang] launching: model={model_path} port={port}")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_file = LOG_PATH.open("ab")
    subprocess.Popen(
        [
            "python", "-m", "sglang.launch_server",
            "--model-path", model_path,
            "--port", str(port),
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if is_running(health_url):
            print(f"[sglang] ready at {health_url} (log: {LOG_PATH})")
            return
        time.sleep(2)
    raise RuntimeError(
        f"[sglang] failed to become healthy within {timeout_s}s; see {LOG_PATH}"
    )


def kill() -> None:
    """Best-effort shutdown of any running SGLang launch_server processes."""
    pkill = shutil.which("pkill")
    if pkill is None:
        print("[sglang] pkill not found; cannot kill server")
        return
    result = subprocess.run(
        [pkill, "-f", "sglang.launch_server"],
        capture_output=True,
    )
    if result.returncode == 0:
        print("[sglang] sent SIGTERM to running launch_server process(es)")
        # give the port time to free up
        for _ in range(15):
            if not is_running():
                break
            time.sleep(1)
    else:
        print("[sglang] no running launch_server process found")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ensure", help="Start SGLang if not running")
    sub.add_parser("kill", help="Stop SGLang if running")
    sub.add_parser("status", help="Print running / not running")
    args = p.parse_args()

    if args.cmd == "ensure":
        ensure_running()
    elif args.cmd == "kill":
        kill()
    elif args.cmd == "status":
        print("running" if is_running() else "not running")
