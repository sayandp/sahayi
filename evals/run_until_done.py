"""Resume answer_eval runs in turns until all finish, waiting out each model's daily window.

    python evals/run_until_done.py evals/results/<run1>.json [evals/results/<run2>.json]

One process and one retriever for all runs: models load once, instead of once per turn in a
fresh process. Each run gets a turn of TURN_ITEMS new items, then the next. A run stopped by
quota (exit 2) sits out for Groq's "try again in" time (at least 5 minutes, at most 2 hours;
30 minutes if no time was given) while the others continue. A run whose turn raises 3 times in
a row is dropped, so real bugs are not retried for days. Memory goes to data/logs/eval_rss.jsonl
after every turn. Exits 1 if any run was dropped, 0 if all finished.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MIN_WAIT_S, MAX_WAIT_S, NO_HINT_WAIT_S = 300, 7200, 1800
TURN_ITEMS = 20
TOTAL_ITEMS = 186
MAX_CRASHES = 3
PEAK_LIMIT_GB = 5.0
RSS_LOG = ROOT / "data" / "logs" / "eval_rss.jsonl"


def stamp() -> str:
    return time.strftime("%F %T")


def finished(run: dict) -> bool:
    items = run["items"]
    unjudged = [
        r
        for r in items
        if not r["abstained"]
        and (r.get("judge_model") != run["judge"] or "error" in (r.get("judge") or {}))
    ]
    return len(items) >= TOTAL_ITEMS and not unjudged


def turn_args(run: dict, path: Path) -> argparse.Namespace:
    """The arguments answer_eval's command line would give for one --all --resume turn."""
    return argparse.Namespace(
        model=run["model"], judge=run["judge"], config=run["config"], all=True, limit=TURN_ITEMS,
        n=10_000, no_answer=10_000, no_eligibility=False, seed=3, live=False, no_judge=False,
        entail=None, resume=str(path), only=None, tag=None,
    )  # fmt: skip


def memory_gb() -> dict[str, float]:
    """Now and peak, in GB. Peak is RSS peak on Windows, high-water RSS elsewhere."""
    import psutil

    info = psutil.Process().memory_info()
    peak = getattr(info, "peak_wset", None)
    if peak is None:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    return {
        "rss_gb": round(info.rss / 1e9, 2),
        "peak_rss_gb": round(peak / 1e9, 2),
        "commit_gb": round(getattr(info, "private", info.vms) / 1e9, 2),
    }


def log_memory(label: str, path: Path = RSS_LOG) -> dict[str, float]:
    mem = memory_gb()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": stamp(), "turn": label, **mem}) + "\n")
    over = f"  OVER {PEAK_LIMIT_GB} GB" if mem["peak_rss_gb"] > PEAK_LIMIT_GB else ""
    print(f"[{stamp()}] memory after {label}: {json.dumps(mem)}{over}", flush=True)
    return mem


def main() -> int:
    from backend.app.answer import load_retriever
    from evals.answer_eval import Pacer
    from evals.answer_eval import run as run_turn

    paths = [Path(p).resolve() for p in sys.argv[1:]]
    ready_at = dict.fromkeys(paths, 0.0)
    crashes = dict.fromkeys(paths, 0)
    live = list(paths)
    dropped = 0
    retriever, pacer = None, Pacer()  # the pacer spans turns: the judge is shared between runs
    try:
        while live:
            progressed = False
            for path in list(live):
                run = json.loads(path.read_text(encoding="utf-8"))
                if finished(run):
                    print(f"[{stamp()}] {path.name}: finished", flush=True)
                    live.remove(path)
                    continue
                if time.time() < ready_at[path]:
                    continue
                if retriever is None:
                    retriever = load_retriever(run["config"])
                print(f"[{stamp()}] {path.name}: turn starts at {len(run['items'])} items",
                      flush=True)  # fmt: skip
                try:
                    code = run_turn(turn_args(run, path), retriever, pacer)
                except Exception:  # noqa: BLE001 - a crashed turn must not end the other run
                    traceback.print_exc()
                    code = 1
                progressed = True
                after = json.loads(path.read_text(encoding="utf-8"))
                print(f"[{stamp()}] {path.name}: exit {code}, {len(after['items'])} items, "
                      f"tokens {json.dumps(after.get('tokens_used'))}", flush=True)  # fmt: skip
                gc.collect()
                log_memory(path.name)
                if code == 2:
                    crashes[path] = 0
                    wait = after.get("retry_after_s") or NO_HINT_WAIT_S
                    wait = min(max(wait, MIN_WAIT_S), MAX_WAIT_S)
                    ready_at[path] = time.time() + wait
                    print(f"[{stamp()}] {path.name}: quota stop, next try in {wait:.0f}s",
                          flush=True)  # fmt: skip
                elif code != 0:
                    crashes[path] += 1
                    if crashes[path] >= MAX_CRASHES:
                        print(f"[{stamp()}] {path.name}: dropped after {MAX_CRASHES} crashes",
                              flush=True)  # fmt: skip
                        live.remove(path)
                        dropped += 1
                else:
                    crashes[path] = 0
            if live and not progressed:
                time.sleep(60)
    finally:
        if retriever is not None:
            retriever.close()
    return 1 if dropped else 0


if __name__ == "__main__":
    sys.exit(main())
