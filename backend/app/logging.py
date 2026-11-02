"""Per-query anonymised trace: tokens, latency, INR cost, retrieved chunk IDs.

No raw question text and no profile facts are logged (CLAUDE.md: no PII). The question
is identified by an HMAC-SHA256 (key LOG_SALT, else random per process) so repeated
questions can be counted without being readable. Records go to the ``sahayi.query``
logger as one JSON object per line and to data/logs/queries.jsonl.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import threading
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from backend.app.config import ROOT, get_settings

LOG_PATH = ROOT / "data" / "logs" / "queries.jsonl"
logger = logging.getLogger("sahayi.query")
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _salt() -> bytes:
    return (get_settings().log_salt or secrets.token_hex(16)).encode()


def question_hash(question: str) -> str:
    return hmac.new(_salt(), question.strip().encode(), hashlib.sha256).hexdigest()[:16]


def trace_record(request_id: str, question: str, retrieved: list[str], result: dict) -> dict:
    meta = result.get("meta", {})
    val = result.get("validation", {})
    return {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "request_id": request_id,
        "q_hash": question_hash(question),
        "q_chars": len(question),
        "lang": result.get("lang"),
        "retrieved": retrieved,
        "cited": [c["chunk_id"] for c in result.get("citations", [])],
        "abstained": result.get("abstained"),
        "abstain_reason": result.get("abstain_reason"),
        "dropped": [d["reason"] for d in val.get("dropped", [])],
        "flags": val.get("flags", []),
        "latency": meta.get("latency"),
        "ttfc_ms": meta.get("ttfc_ms"),
        "ttft_ms": meta.get("ttft_ms"),
        "tokens_in": meta.get("tokens_in"),
        "tokens_out": meta.get("tokens_out"),
        "cost_inr": meta.get("cost_inr"),
        "model": meta.get("model"),
        "rerank_backend": meta.get("rerank_backend"),
        "rerank_fallback": meta.get("rerank_fallback"),
    }


def log_trace(record: dict, path: Path | None = LOG_PATH) -> None:
    line = json.dumps(record, ensure_ascii=False)
    logger.info(line)
    if path is None:
        return
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class JsonFormatter(logging.Formatter):
    """Structured app logs: one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        if record.name == "sahayi.query":
            return record.getMessage()
        out = {"ts": datetime.now(UTC).isoformat(timespec="seconds"), "level": record.levelname,
               "logger": record.name, "msg": record.getMessage()}  # fmt: skip
        for key in ("request_id", "path", "status", "ms", "client"):
            if hasattr(record, key):
                out[key] = getattr(record, key)
        return json.dumps(out, ensure_ascii=False)


def configure_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    for name in ("sahayi", "sahayi.query"):
        log = logging.getLogger(name)
        log.handlers[:] = [handler]
        log.setLevel(level)
        log.propagate = False
