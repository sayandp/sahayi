"""Assemble the Hugging Face Space folder: code, frontend source, index, chunks, int8 models.

    python deploy/build_bundle.py [--out data/deploy_space]

Reads no keys and copies no .env. The result is what gets pushed to the Space, and also what
deploy/bench_retrieval.py runs from, so the benchmark sees exactly the shipped files.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JINA = "models--jinaai--jina-reranker-v2-base-multilingual"
COLLECTION = "chunks_section_aware_bge-m3_ctx"
SPACE_README = """---
title: Sahayi
emoji: 📄
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
---

Sahayi: which Kerala government schemes can you get? Every answer cites its source document.
"""
GITATTRIBUTES = (
    "*.onnx filter=lfs diff=lfs merge=lfs -text\n*.sqlite filter=lfs diff=lfs merge=lfs -text\n"
)
SKIP = shutil.ignore_patterns("__pycache__", "node_modules", "dist", "screenshots", "test-results",
                              "playwright-report", "tests", ".env", ".env.*", "*.pyc")  # fmt: skip


def tokenizer_files(folder: Path) -> list[Path]:
    names = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "sentencepiece.bpe.model", "config.json")  # fmt: skip
    return [folder / n for n in names if (folder / n).exists()]


def copy_files(files: list[Path], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copyfile(f, dest / f.name)  # follows the HF cache's links


def build(out: Path) -> dict[str, int]:
    out.mkdir(parents=True, exist_ok=True)
    for child in out.iterdir():  # not the folder itself: a shell may be sitting in it
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    (out / "README.md").write_text(SPACE_README, encoding="utf-8")
    (out / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8", newline="\n")
    shutil.copyfile(ROOT / "deploy" / "Dockerfile", out / "Dockerfile")
    for name in ("backend", "deploy", "frontend"):
        shutil.copytree(ROOT / name, out / name, ignore=SKIP)
    # retrieval tokenises with ingest.clean; the rest of ingest/ is not needed at runtime
    names = ("__init__.py", "clean.py", "common.py", "models.py", "sources.yaml")
    copy_files([ROOT / "ingest" / n for n in names], out / "ingest")
    copy_files([ROOT / "data/processed/chunks.section_aware.jsonl"], out / "data/processed")
    copy_files([ROOT / "data/raw/manifest.jsonl"], out / "data/raw")
    copy_files([ROOT / "data/index/bm25.section_aware.ctx.pkl"], out / "data/index")

    qdrant, target = ROOT / "data/index/qdrant", out / "data/index/qdrant"
    meta = json.loads((qdrant / "meta.json").read_text(encoding="utf-8"))
    meta["collections"] = {COLLECTION: meta["collections"][COLLECTION]}
    target.mkdir(parents=True)
    (target / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    shutil.copytree(qdrant / "collection" / COLLECTION, target / "collection" / COLLECTION)

    onnx = out / "data/cache/onnx"
    query = ROOT / "data/cache/onnx/bge-m3-query"
    copy_files([query / "model_int8.onnx", *tokenizer_files(query)], onnx / "bge-m3-query")
    snap = next((ROOT / "data/cache/hf/hub" / JINA / "snapshots").iterdir())
    copy_files([snap / "onnx/model_int8.onnx", *tokenizer_files(snap)], onnx / "jina-reranker-v2")

    def size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    return {
        "total": size(out),
        "bge-m3 query encoder": size(onnx / "bge-m3-query"),
        "jina reranker": size(onnx / "jina-reranker-v2"),
        "index and chunks": size(out / "data/index") + size(out / "data/processed"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "deploy_space"))
    args = ap.parse_args()
    for name, n in build(Path(args.out)).items():
        print(f"{name}: {n / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
