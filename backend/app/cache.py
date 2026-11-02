"""Answer cache for the API: normalised question + language + corpus version, 7-day TTL.

Only general questions are cached. Anything where the user states facts about themselves
(first-person words, any number, an eligibility verdict, a follow-up question) is never
written, so no profile data reaches the disk. Keys are hashes; the question is not stored.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

from backend.app.llm.prompts import ANSWER_VERSION

_PERSONAL = re.compile(
    r"\d|[൦-൯]|\b(?:i|i'm|im|me|my|mine|we|our|us|am|enikk?u|ente|njan|nja?ngal|njangalude|"
    r"amma|achan|bharthavu?|bharya|makan|makal)\b|"
    r"ഞാൻ|ഞങ്ങ|എനിക്ക|എന്റെ|എന്നെ|അമ്മ|അച്ഛൻ|ഭർത്താവ|ഭാര്യ|മകൻ|മകൾ|മകന|മകള",
    re.I,
)


def normalise(question: str) -> str:
    text = unicodedata.normalize("NFC", question).casefold()
    return re.sub(r"\s+", " ", text).strip(" \t\n?.!।,")


def is_personal(question: str) -> bool:
    return bool(_PERSONAL.search(question))


class AnswerCache:
    def __init__(self, directory: Path, corpus_version: str, ttl_days: float = 7) -> None:
        self.dir, self.corpus, self.ttl = directory, corpus_version, ttl_days * 86400

    def _path(self, question: str, lang: str) -> Path:
        raw = "\x00".join([ANSWER_VERSION, self.corpus, lang, normalise(question)])
        return self.dir / f"{hashlib.sha256(raw.encode()).hexdigest()}.json"

    def get(self, question: str, lang: str) -> dict | None:
        path = self._path(question, lang)
        if not path.exists():
            return None
        entry = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - entry["ts"] > self.ttl:
            path.unlink(missing_ok=True)
            return None
        return entry

    def put(self, question: str, lang: str, sources: dict, result: dict) -> bool:
        """Store a finished general answer; returns whether it was stored."""
        if (
            is_personal(question)
            or result.get("degraded")
            or result.get("abstain_reason") == "error"
            or result.get("follow_up_question")
            or any(card["eligible"] is not None for card in result["schemes"])
        ):
            return False
        path = self._path(question, lang)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"ts": time.time(), "sources": sources, "result": result}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
        return True
