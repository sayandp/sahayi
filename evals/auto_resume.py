"""Task Scheduler entry point: resume the Gate 2 runs at logon and on wake, one copy only.

    python evals/auto_resume.py evals/results/<run1>.json evals/results/<run2>.json
    python evals/auto_resume.py --check <runs...>   # says what it would do; starts nothing

Exits at once if another copy holds the lock, or if a run_until_done or answer_eval started
some other way is already running (local Qdrant allows one process). The lock is an OS file
lock, so a crash or power loss cannot leave it stuck. Log: data/logs/auto_resume.log.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "data" / "cache" / "tmp" / "gate2_auto_resume.lock"
LOG = ROOT / "data" / "logs" / "auto_resume.log"
DRIVERS = ("run_until_done.py", "answer_eval.py", "gate2_finish.py", "judge_agreement.py")
MAX_STARTS, RESTART_WAIT_S = 3, 300
CHILD_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}  # the log file must take Malayalam


def acquire_lock(path: Path) -> IO | None:
    """An open file holding an exclusive lock, or None if another process has it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        if sys.platform == "win32":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return None
    return handle


def driver_running(cmdlines: Iterable[list[str]]) -> bool:
    return any(any(part.replace("\\", "/").endswith(DRIVERS) for part in c) for c in cmdlines)


def process_cmdlines() -> list[list[str]]:
    import psutil

    out = []
    for proc in psutil.process_iter(["name", "cmdline"]):
        if "python" in (proc.info["name"] or "").lower() and proc.info["cmdline"]:
            out.append(proc.info["cmdline"])
    return out


def main() -> int:
    args = sys.argv[1:]
    check = "--check" in args
    runs = [str((ROOT / a).resolve()) for a in args if a != "--check"]
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as log:

        def say(text: str) -> None:
            print(f"[{time.strftime('%F %T')}] {text}", file=log, flush=True)
            if check:
                print(text)

        lock = acquire_lock(LOCK)
        if lock is None:
            say("skip: another auto_resume holds the lock")
            return 0
        if driver_running(process_cmdlines()):
            say("skip: an eval driver is already running")
            return 0
        if check:
            say(f"would start run_until_done on {len(runs)} run files")
            return 0
        python = sys.executable.replace("pythonw", "python")
        driver = [python, str(ROOT / "evals" / "run_until_done.py"), *runs]
        code = 1
        for attempt in range(1, MAX_STARTS + 1):
            say(f"starting run_until_done (attempt {attempt}/{MAX_STARTS})")
            code = subprocess.run(
                driver, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV
            ).returncode
            say(f"run_until_done exited {code}")
            if code == 0:
                break
            if attempt < MAX_STARTS:  # a native crash takes the process down: back off, retry
                time.sleep(RESTART_WAIT_S)
        if code == 0:
            say("both runs finished: starting gate2_finish")
            finish = [python, str(ROOT / "evals" / "gate2_finish.py"), *runs]
            code = subprocess.run(
                finish, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=CHILD_ENV
            ).returncode
            say(f"gate2_finish exited {code}")
        lock.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
