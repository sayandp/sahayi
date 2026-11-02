"""Task runner that works without make (Windows): python tasks.py <task> [args...]

Tasks:
  setup    install Python deps, the Playwright Chromium build and the pre-commit hook
  ingest   download -> extract -> clean -> chunk (all strategies) -> report
  report   rebuild data/processed/ingest_report.md from existing outputs
  index    build BM25 + Qdrant indexes (python -m ingest.build_index)
  gold     resolve evals/gold/questions_src.yaml -> questions.jsonl
  eval     retrieval eval (default config; pass config names to compare)
  answer-eval  answer eval / Gate 2 (evals/answer_eval.py; extra args passed through)
  serve    FastAPI app on :8077 (uvicorn backend.app.main:app; API_PORT overrides)
  up       build the UI, start the app with full answers, open the browser
           (--retrieval-only: no LLM calls; --no-browser)
  down     stop the app started by up or share
  share    public demo link through a Cloudflare quick tunnel, until Ctrl+C
  test     pytest (offline)
  lint     ruff + black --check
  fmt      ruff --fix + black

Extra arguments are passed to the underlying command for test/lint/fmt, and to
`ingest.extract` for ingest (e.g. `python tasks.py ingest --no-ocr`).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
CODE_DIRS = ["ingest", "backend", "evals", "tasks.py"]
PID_FILE = ROOT / "data" / "run" / "api.pid.json"
API_LOG = ROOT / "data" / "logs" / "api.log"
TUNNEL_LOG = ROOT / "data" / "logs" / "tunnel.log"
SERVE_INDEX = ROOT / "data" / "index" / "qdrant_serve"
CLOUDFLARED = ROOT / "tools" / "cloudflared.exe"
CLOUDFLARED_RELEASE = "https://api.github.com/repos/cloudflare/cloudflared/releases/latest"
CLOUDFLARED_ASSET = "cloudflared-windows-amd64.exe"
TUNNEL_URL = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
START_TIMEOUT_S, TUNNEL_TIMEOUT_S = 240, 60


def run(*cmd: str) -> None:
    print("$", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode:
        sys.exit(result.returncode)


def setup(args: list[str]) -> None:
    run(PY, "-m", "pip", "install", "-r", "backend/requirements.txt")
    run(PY, "-m", "playwright", "install", "chromium")
    run("git", "config", "core.hooksPath", ".githooks")


def ingest(args: list[str]) -> None:
    run(PY, "-m", "ingest.download")
    run(PY, "-m", "ingest.extract", *args)
    run(PY, "-m", "ingest.clean")
    run(PY, "-m", "ingest.chunk", "--strategy", "all")
    report(args)


def report(args: list[str]) -> None:
    run(PY, "-m", "ingest.report")


def index(args: list[str]) -> None:
    run(PY, "-m", "ingest.build_index", *args)


def gold(args: list[str]) -> None:
    run(PY, "evals/build_gold.py")


def eval_(args: list[str]) -> None:
    run(PY, "evals/run_eval.py", "--config", *(args or ["default"]))


def answer_eval(args: list[str]) -> None:
    run(PY, "evals/answer_eval.py", *args)


def serve(args: list[str]) -> None:
    from backend.app.config import get_settings

    port = str(get_settings().api_port)
    run(PY, "-m", "uvicorn", "backend.app.main:app", "--port", port, *args)


def build_frontend() -> None:
    npm = shutil.which("npm")
    if npm is None:
        sys.exit("npm not found: install Node.js 22 first")
    ui = ROOT / "frontend"
    if not (ui / "node_modules").exists():
        subprocess.run([npm, "ci"], cwd=ui, check=True)
    subprocess.run(
        [npm, "run", "build"], cwd=ui, check=True, env={**os.environ, "VITE_API_URL": "/"}
    )


def serve_index() -> Path:
    """The API's own copy of the index: local Qdrant locks its folder, and evals hold that one."""
    from backend.app.config import get_settings
    from backend.app.retrieval.search import load_named_config

    collection = load_named_config(get_settings().retrieval_config).collection
    source, target = ROOT / "data" / "index" / "qdrant", SERVE_INDEX
    meta = target / "meta.json"
    fresh = meta.exists() and meta.stat().st_mtime >= (source / "meta.json").stat().st_mtime
    if fresh and (target / "collection" / collection).exists():
        return target
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    info = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    info["collections"] = {collection: info["collections"][collection]}
    shutil.copytree(source / "collection" / collection, target / "collection" / collection)
    meta.write_text(json.dumps(info), encoding="utf-8")
    return target


def api_url() -> str:
    from backend.app.config import get_settings

    return f"http://127.0.0.1:{get_settings().api_port}"


def health() -> dict | None:
    try:
        with urllib.request.urlopen(f"{api_url()}/health", timeout=3) as res:
            return json.load(res)
    except (OSError, ValueError):
        return None


def running_api():  # noqa: ANN201 - psutil.Process or None
    """The process in the pid file, if it is still our server (Windows reuses pids)."""
    import psutil

    if not PID_FILE.exists():
        return None
    try:
        proc = psutil.Process(json.loads(PID_FILE.read_text(encoding="utf-8"))["pid"])
        return proc if "backend.app.main:app" in " ".join(proc.cmdline()) else None
    except (psutil.Error, ValueError, KeyError):
        return None


def start_api(env: dict[str, str]) -> None:
    """Start the server detached, wait until the models are loaded and /health answers."""
    from backend.app.config import get_settings

    if health() is not None:
        sys.exit(f"something already answers on {api_url()}: run `python tasks.py down` first")
    API_LOG.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    full_env = {**os.environ, "STATIC_DIR": str(ROOT / "frontend" / "dist"),
                "SERVE_QDRANT_PATH": str(serve_index()), "ANSWER_MODE": "generate",
                "DEMO_MODE": "0", "PYTHONIOENCODING": "utf-8", **env}  # fmt: skip
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    cmd = [PY, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1",
           "--port", str(get_settings().api_port)]  # fmt: skip
    with API_LOG.open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=full_env, stdout=log,
                                stderr=subprocess.STDOUT, creationflags=flags)  # fmt: skip
    PID_FILE.write_text(json.dumps({"pid": proc.pid}), encoding="utf-8")
    t0 = time.monotonic()
    print("loading models...", flush=True)
    while time.monotonic() - t0 < START_TIMEOUT_S:
        if proc.poll() is not None:
            PID_FILE.unlink(missing_ok=True)
            sys.exit(f"the server exited with code {proc.returncode}; see {API_LOG}")
        state = health()
        if state and state["status"] == "ok":
            print(f"ready in {time.monotonic() - t0:.1f} s", flush=True)
            return
        time.sleep(1)
    stop_api()
    sys.exit(f"the server did not answer within {START_TIMEOUT_S} s; see {API_LOG}")


def stop_tunnel() -> bool:
    """End any cloudflared started from tools/, even if `share` itself was killed."""
    import psutil

    stopped = False
    for proc in psutil.process_iter(["exe"]):
        if proc.info["exe"] and Path(proc.info["exe"]) == CLOUDFLARED:
            proc.terminate()
            stopped = True
    return stopped


def stop_api() -> bool:
    import psutil

    proc = running_api()
    PID_FILE.unlink(missing_ok=True)
    if proc is None:
        return False
    family = [*proc.children(recursive=True), proc]
    for p in family:
        p.terminate()
    _, alive = psutil.wait_procs(family, timeout=15)
    for p in alive:
        p.kill()
    return True


def up(args: list[str]) -> None:
    """Build the UI, start the API with full answers on API_PORT, open the browser.

    --retrieval-only: no LLM calls (sources and a busy notice).  --no-browser: do not open it.
    """
    if running_api() is not None:
        print(f"already running: {api_url()}")
    else:
        build_frontend()
        env = {"ANSWER_MODE": "retrieval_only"} if "--retrieval-only" in args else {}
        start_api(env)
        print(f"Sahayi is running: {api_url()}  (stop it with: python tasks.py down)")
    if "--no-browser" not in args:
        webbrowser.open(api_url())


def down(args: list[str]) -> None:
    tunnel, api = stop_tunnel(), stop_api()
    print("stopped" if tunnel or api else "not running")


def verified_cloudflared() -> Path:
    """Download cloudflared to tools/ once. Refuses a file that fails the published checksum."""
    if CLOUDFLARED.exists():
        return CLOUDFLARED
    if sys.platform != "win32":
        found = shutil.which("cloudflared")
        if found:
            return Path(found)
        sys.exit("install cloudflared first: this downloader only handles Windows")
    import httpx

    release = httpx.get(CLOUDFLARED_RELEASE, timeout=30, follow_redirects=True).json()
    asset = next(a for a in release["assets"] if a["name"] == CLOUDFLARED_ASSET)
    match = re.search(
        rf"{re.escape(CLOUDFLARED_ASSET)}:\s*([0-9a-f]{{64}})", release.get("body", "")
    )
    print(f"downloading cloudflared {release['tag_name']} to {CLOUDFLARED}", flush=True)
    data = httpx.get(asset["browser_download_url"], timeout=300, follow_redirects=True).content
    digest = hashlib.sha256(data).hexdigest()
    if match is None:
        sys.exit(f"the release notes list no checksum for {CLOUDFLARED_ASSET}; not installing")
    if digest != match.group(1):
        sys.exit("cloudflared checksum mismatch; not installing")
    CLOUDFLARED.parent.mkdir(parents=True, exist_ok=True)
    CLOUDFLARED.write_bytes(data)
    print(f"checksum ok: {digest}", flush=True)
    return CLOUDFLARED


def share(args: list[str]) -> None:
    """Public demo link through a Cloudflare quick tunnel, until Ctrl+C. See README, Demo link."""
    from backend.app.config import get_settings

    binary = verified_cloudflared()
    stop_api()
    build_frontend()
    start_api({"DEMO_MODE": "1"})
    TUNNEL_LOG.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(signal, "SIGBREAK"):  # closing the console window or Ctrl+Break
        signal.signal(signal.SIGBREAK, signal.default_int_handler)
    tunnel = subprocess.Popen(
        [str(binary), "tunnel", "--no-autoupdate", "--url", api_url()],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
        errors="replace", bufsize=1,
    )  # fmt: skip
    found = threading.Event()

    def read_output() -> None:
        with TUNNEL_LOG.open("w", encoding="utf-8") as log:
            for line in tunnel.stdout:
                log.write(line)
                log.flush()
                url = TUNNEL_URL.search(line)
                if url and not found.is_set():
                    found.set()
                    s = get_settings()
                    print(f"\nPUBLIC DEMO LINK: {url.group(0)}\n"
                          f"Limits: {s.demo_rate_limit_per_min}/min per visitor, "
                          f"{s.demo_daily_cap} generated answers a day, then busy mode.\n"
                          "Anyone with this link can use your Groq quota until you stop it.\n"
                          "Press Ctrl+C to stop.\n", flush=True)  # fmt: skip

    threading.Thread(target=read_output, daemon=True).start()
    try:
        t0 = time.monotonic()
        while tunnel.poll() is None:
            if not found.is_set() and time.monotonic() - t0 > TUNNEL_TIMEOUT_S:
                print(f"no tunnel URL after {TUNNEL_TIMEOUT_S} s; see {TUNNEL_LOG}", flush=True)
                break
            time.sleep(0.5)
        else:
            print(f"the tunnel exited with code {tunnel.returncode}; see {TUNNEL_LOG}", flush=True)
    except KeyboardInterrupt:
        print("\nstopping...", flush=True)
    finally:
        if tunnel.poll() is None:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel.kill()
        stop_api()
        print("demo link closed and server stopped")


def test(args: list[str]) -> None:
    run(PY, "-m", "pytest", "backend/tests", *args)


def lint(args: list[str]) -> None:
    run(PY, "-m", "ruff", "check", *CODE_DIRS, *args)
    run(PY, "-m", "black", "--check", *CODE_DIRS)


def fmt(args: list[str]) -> None:
    run(PY, "-m", "ruff", "check", "--fix", *CODE_DIRS, *args)
    run(PY, "-m", "black", *CODE_DIRS)


TASKS: dict[str, Callable[[list[str]], None]] = {
    "setup": setup,
    "ingest": ingest,
    "report": report,
    "index": index,
    "gold": gold,
    "eval": eval_,
    "answer-eval": answer_eval,
    "serve": serve,
    "up": up,
    "down": down,
    "share": share,
    "test": test,
    "lint": lint,
    "fmt": fmt,
}


def main(argv: list[str]) -> None:
    if not argv or argv[0] not in TASKS:
        print(__doc__)
        sys.exit(2)
    TASKS[argv[0]](argv[1:])


if __name__ == "__main__":
    main(sys.argv[1:])
