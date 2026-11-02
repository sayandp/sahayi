"""Serve the Space folder on :7860 in retrieval-only mode, pinned to 2 cores, and run Playwright.

    python deploy/smoke_local.py

The stand-in for `docker run` on a machine without Docker: same files and same start command,
but on Windows, in a torch-free venv. No API key is passed, so no LLM can be called.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "data" / "deploy_space"
VENV_PY = ROOT / "data" / "deploy_venv" / "Scripts" / "python.exe"
URL = "http://127.0.0.1:7860"
NPX = shutil.which("npx.cmd") or shutil.which("npx") or "npx"


def wait_ready(proc: subprocess.Popen, t0: float, limit: float = 300) -> float:
    while time.perf_counter() - t0 < limit:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{URL}/health", timeout=2) as r:
                if r.status == 200:
                    return time.perf_counter() - t0
        except OSError:
            time.sleep(0.2)
    raise SystemExit("server not ready in time")


def main() -> int:
    tmp = str(ROOT / "data/cache/tmp")
    ui_env = {**os.environ, "VITE_API_URL": "/", "TMP": tmp, "TEMP": tmp,
              "npm_config_cache": str(ROOT / "data/cache/npm")}  # fmt: skip
    subprocess.run([NPX, "vite", "build", "--outDir", str(SPACE / "ui"), "--emptyOutDir"],
                   cwd=ROOT / "frontend", env=ui_env, check=True, shell=False)  # fmt: skip
    env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    env |= {"ANSWER_MODE": "retrieval_only", "STATIC_DIR": str(SPACE / "ui"), "PORT": "7860",
            "RETRIEVAL_CONFIGS_PATH": str(SPACE / "deploy/configs.yaml"),
            "RETRIEVAL_CONFIG": "deploy_local", "HF_HUB_OFFLINE": "1", "TMP": tmp, "TEMP": tmp,
            "SAHAYI_RERANK_CACHE": "0", "SAHAYI_REWRITE_CACHE": "0",
            "PYTHONIOENCODING": "utf-8"}  # fmt: skip
    cold = []
    results: dict = {}
    for attempt in range(2):
        t0 = time.perf_counter()
        log = (ROOT / f"data/cache/tmp/p4b_server{attempt}.log").open("w", encoding="utf-8")
        proc = subprocess.Popen([str(VENV_PY), "deploy/serve.py", "--cpus", "2"], cwd=SPACE,
                                env=env, stdout=log, stderr=subprocess.STDOUT)  # fmt: skip
        try:
            cold.append(round(wait_ready(proc, t0), 1))
            if attempt == 1:  # the second start: the first only warms the OS file cache
                e2e = {**ui_env, "E2E_BASE_URL": URL,
                       "PLAYWRIGHT_BROWSERS_PATH": str(ROOT / "data/cache/playwright")}  # fmt: skip
                results["playwright_exit"] = subprocess.run(
                    [NPX, "playwright", "test"], cwd=ROOT / "frontend", env=e2e
                ).returncode
                mem = psutil.Process(proc.pid).memory_info()
                results["server_peak_ram_gb"] = round(getattr(mem, "peak_wset", mem.rss) / 1e9, 2)
                with urllib.request.urlopen(f"{URL}/health", timeout=5) as r:
                    results["health"] = json.load(r)
        finally:
            proc.terminate()
            proc.wait(30)
            log.close()
        time.sleep(3)
    results["cold_start_to_health_s"] = {"first": cold[0], "second": cold[1]}
    print(json.dumps(results, indent=1))
    return results["playwright_exit"]


if __name__ == "__main__":
    sys.exit(main())
