"""pydantic-settings configuration loaded from env vars and the repo-root .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # OCR. Unset/empty -> look up `tesseract` on PATH.
    tesseract_cmd: str | None = None

    # Caches live on the repo drive (gitignored data/cache/) so multi-GB models and
    # browsers never fill the system drive. Set here, not only in the shell, so code
    # behaves the same whatever environment it is started from.
    hf_home: Path = ROOT / "data" / "cache" / "hf"
    playwright_browsers_path: Path = ROOT / "data" / "cache" / "playwright"
    tmp_dir: Path = ROOT / "data" / "cache" / "tmp"  # ONNX export, Chromium profiles

    @property
    def hf_hub_cache(self) -> Path:
        return self.hf_home / "hub"

    # Later phases (see CLAUDE.md).
    groq_api_key: str = ""
    gemini_api_key: str = ""
    llm_primary: str = "groq"
    # Groq retired its Llama chat models (checked 2026-10-20: gpt-oss-20b/120b, qwen3.8-27b).
    llm_model_groq: str = "openai/gpt-oss-120b"
    llm_model_gemini: str = "gemini-2.0-flash"
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    qdrant_path: str = "./data/index/qdrant"
    top_k_retrieve: int = 20
    top_k_rerank: int = 5

    # Phase 3: answering + API.
    # Groq generator: qwen passed Gate 2 with the lowest TTFT of the two candidates
    # benchmarked (evals/results/phase3_summary.md); it also does the query rewrite
    answer_model: str = "qwen/qwen3.8-27b"
    # Production fallback chain: answer_model -> these Groq models -> answer_model_gemini.
    # Checked 2026-10-30: gemini-2.0-flash is shut down; on this key gemini-3.8-flash and
    # 3.7-flash hung past 45 s while 3.6-flash answered in about 5 s.
    answer_fallback_models: list[str] = ["openai/gpt-oss-20b"]
    answer_model_gemini: str = "gemini-3.6-flash"
    answer_cache_days: int = 7
    api_port: int = 8077
    # Deploy image only; the empty / default values change nothing elsewhere.
    retrieval_configs_path: str = ""  # YAML with retrieval_config; empty: evals/configs.yaml
    answer_mode: str = "generate"  # or retrieval_only: no LLM calls, sources + cards + notice
    static_dir: str = ""  # built frontend to serve from FastAPI
    serve_qdrant_path: str = ""  # the API's own index copy (tasks.py up), so evals can run too
    # Demo link (tasks.py share). Off everywhere else.
    demo_mode: bool = False
    demo_rate_limit_per_min: int = 5  # per visitor IP
    demo_daily_cap: int = 100  # generated answers per day; cached answers do not count
    retrieval_config: str = "default"  # evals/configs.yaml entry; "deploy" = hosted rerank
    jina_api_key: str = ""
    cohere_api_key: str = ""
    rate_limit_per_min: int = 10  # /ask + /ask/sync requests per client IP
    log_salt: str = ""  # HMAC key for question hashes in logs; empty -> random per process
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    # Groq on-demand prices, USD per 1M tokens (input, output). Checked 2026-10-20 on
    # https://console.groq.com/docs/models. Hosted Jina rerank is on the free key tier.
    llm_prices_usd_per_m: dict[str, tuple[float, float]] = {
        "openai/gpt-oss-20b": (0.075, 0.30),
        "openai/gpt-oss-120b": (0.15, 0.60),
        "qwen/qwen3.8-27b": (0.80, 4.00),
        "gemini-3.6-flash": (0.75, 3.75),  # paid rate to 2026-12-31 (checked 2026-10-30)
    }
    prices_checked: str = "2026-10-20"
    usd_inr: float = 96.5  # checked 2026-10-20

    def cost_inr(self, model: str, tokens_in: int, tokens_out: int) -> float:
        p_in, p_out = self.llm_prices_usd_per_m.get(model, (0.0, 0.0))
        return (tokens_in * p_in + tokens_out * p_out) / 1e6 * self.usd_inr


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_tmp() -> Path:
    """Send this process's temp files (and its children's) to the repo drive."""
    import os
    import tempfile

    path = get_settings().tmp_dir
    path.mkdir(parents=True, exist_ok=True)
    for name in ("TMP", "TEMP", "TMPDIR"):
        os.environ[name] = str(path)
    tempfile.tempdir = str(path)
    return path


def configure_hf_cache() -> Path:
    """Point huggingface_hub at the project cache (call before loading models)."""
    import os

    configure_tmp()
    settings = get_settings()
    path = settings.hf_hub_cache
    path.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(settings.hf_home)
    os.environ["HF_HUB_CACHE"] = str(path)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return path


def configure_playwright_browsers() -> Path:
    """Point Playwright at the project browser cache (call before launching)."""
    import os

    configure_tmp()
    path = get_settings().playwright_browsers_path
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(path)
    return path
