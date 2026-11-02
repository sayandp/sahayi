PY ?= python

.PHONY: setup ingest index gold eval answer-eval serve test lint fmt

setup:
	$(PY) -m pip install -r backend/requirements.txt
	$(PY) -m playwright install chromium
	if [ -f frontend/package.json ]; then cd frontend && npm i; fi

# Phase 1: download -> extract -> clean -> chunk (all strategies) -> report.
# build_index is Phase 2 and is not run yet.
ingest:
	$(PY) -m ingest.download
	$(PY) -m ingest.extract
	$(PY) -m ingest.clean
	$(PY) -m ingest.chunk --strategy all
	$(PY) -m ingest.report

index:
	$(PY) -m ingest.build_index

gold:
	$(PY) evals/build_gold.py

eval:
	$(PY) evals/run_eval.py --config default

answer-eval:
	$(PY) evals/answer_eval.py

serve:
	$(PY) -m uvicorn backend.app.main:app --reload --port 8077

test:
	$(PY) -m pytest backend/tests

lint:
	$(PY) -m ruff check ingest backend evals tasks.py
	$(PY) -m black --check ingest backend evals tasks.py

fmt:
	$(PY) -m ruff check --fix ingest backend evals tasks.py
	$(PY) -m black ingest backend evals tasks.py
