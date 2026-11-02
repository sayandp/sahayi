"""Structured user profile from the query rewrite's ``profile_hints``.

The rewrite call (query.py) already extracts the facts a user states about themselves;
this module only whitelists and normalises them for the answer prompt. Profiles are
used for the one request and never stored or logged (CLAUDE.md: no PII).
"""

from __future__ import annotations

import json

FIELDS = ("age", "gender", "district", "occupation", "income_band", "ration_card", "category")


def clean_profile(hints: dict | None) -> dict:
    out: dict = {}
    for key in FIELDS:
        value = (hints or {}).get(key)
        if value in (None, "", "null", "unknown"):
            continue
        if key == "age":
            try:
                value = int(str(value).split()[0])
            except ValueError:
                continue
            if not 0 < value < 120:
                continue
        out[key] = value if isinstance(value, int) else str(value)[:80]
    return out


def profile_text(profile: dict) -> str:
    """Prompt text; with no hints the model reads the facts from the question itself."""
    if not profile:
        return "none extracted (use only facts stated in the question)"
    return json.dumps(profile, ensure_ascii=False)
