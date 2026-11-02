"""After both Gate 2 runs finish: live latency, Gemini agreement, final report, commit, push.

    python evals/gate2_finish.py evals/results/<run1>.json evals/results/<run2>.json

Safe to run again: each step looks at its output file and continues or skips. Steps that stop
on quota wait and retry. Gemini's free tier allows about 20 requests a day, so the agreement
check for the second run may take until the next day.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evals" / "results"
LIVE_ITEMS, AGREEMENT_N = 30, 15
SECOND_JUDGE = "gemini-3.6-flash"
QUOTA_WAIT_S, MAX_TRIES = 1800, 96
COMMIT_MESSAGE = """Add final Gate 2 results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
"""


def live_tag(model: str) -> str:
    return "latency_" + model.split("/")[-1].replace(".", "-")


def run_step(cmd: list[str]) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT).returncode


def live_run(model: str, judge: str) -> None:
    """Exactly LIVE_ITEMS items, across as many quota stops as it takes."""
    tag = live_tag(model)
    base = [sys.executable, "evals/answer_eval.py", "--all", "--live", "--no-judge",
            "--model", model, "--judge", judge]  # fmt: skip
    for _ in range(MAX_TRIES):
        files = sorted(RESULTS.glob(f"*_answers_{tag}.json"))
        have = len(json.loads(files[-1].read_text("utf-8"))["items"]) if files else 0
        if have >= LIVE_ITEMS:
            return
        extra = ["--resume", str(files[-1])] if files else ["--tag", tag]
        code = run_step([*base, "--limit", str(LIVE_ITEMS - have), *extra])
        if code not in (0, 2):
            raise SystemExit(f"live run for {model} exited {code}")
        if code == 2:
            time.sleep(QUOTA_WAIT_S)
    raise SystemExit(f"live run for {model} did not finish")


def agreement(run: Path) -> None:
    if run.with_name(run.stem + "_judge_agreement.json").exists():
        return
    cmd = [sys.executable, "evals/judge_agreement.py", str(run), "--second", SECOND_JUDGE,
           "--n", str(AGREEMENT_N)]  # fmt: skip
    if run_step(cmd) != 0:
        raise SystemExit(f"judge agreement for {run.name} failed")


def publish() -> int:
    """Commit only the results and the README block, then push. Never forces."""
    if run_step(["git", "add", "README.md", "evals/results"]) != 0:
        return 1
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("nothing new to commit", flush=True)
        return 0
    if run_step(["git", "commit", "-m", COMMIT_MESSAGE]) != 0:
        return 1
    return run_step(["git", "push", "origin", "main"])


def main() -> int:
    runs = [Path(p).resolve() for p in sys.argv[1:]]
    for run in runs:
        data = json.loads(run.read_text("utf-8"))
        live_run(data["model"], data["judge"])
    for run in runs:
        agreement(run)
    code = run_step([sys.executable, "evals/gate2_report.py", *map(str, runs)])
    return code or publish()


if __name__ == "__main__":
    sys.exit(main())
