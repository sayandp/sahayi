"""Recompute the summary of an answer_eval run JSON with the current metric code.

python evals/resummarize.py RUN.json [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.answer_eval import summarize  # noqa: E402

for path in sys.argv[1:]:
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    run["summary"] = summarize(run["items"])
    Path(path).write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"resummarized {path}")
