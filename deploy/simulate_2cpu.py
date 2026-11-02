"""Run the retrieval benchmark from the Space folder, pinned to 2 cores (no container needed).

    python deploy/simulate_2cpu.py

Local, hosted, then local again to expose drift. Only Jina's key is passed, and only to the hosted
run; the Space folder has no .env, so Groq and Gemini cannot be called. Results: evals/results/.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "data" / "deploy_space"
VENV_PY = ROOT / "data" / "deploy_venv" / "Scripts" / "python.exe"
RUNS = [("deploy_local", "local-1"), ("deploy_hosted", "hosted"), ("deploy_local", "local-2")]


def main() -> int:
    shutil.copytree(ROOT / "data/cache/rewrite", SPACE / "data/cache/rewrite", dirs_exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
    base |= {"PYTHONIOENCODING": "utf-8", "SAHAYI_REWRITE_CACHE": "1",
             "TMP": str(ROOT / "data/cache/tmp"), "TEMP": str(ROOT / "data/cache/tmp")}  # fmt: skip
    summaries = {}
    for config, label in RUNS:
        env = dict(base)
        if config == "deploy_hosted":
            env["JINA_API_KEY"] = dotenv_values(ROOT / ".env").get("JINA_API_KEY") or ""
        out = ROOT / "evals" / "results" / f"{stamp}_deploy2cpu_{label}.json"
        code = subprocess.run(
            [str(VENV_PY), "deploy/bench_retrieval.py", "--config", config, "--cpus", "2",
             "--gold", str(ROOT / "evals/gold/questions.jsonl"), "--out", str(out)],
            cwd=SPACE, env=env,
        ).returncode  # fmt: skip
        if code:
            print(f"{label}: exit {code}", flush=True)
            return code
        summaries[label] = json.loads(out.read_text(encoding="utf-8"))["summary"]
    print(json.dumps(summaries, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
