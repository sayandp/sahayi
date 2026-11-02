"""Dense retrieval: sentence-transformers embeddings stored in local Qdrant.

Embeddings are cached on disk by (model, sha1(text)), so re-indexing after a
chunking or config change only embeds new texts.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

# Models that expect instruction prefixes (e5 family); bge-m3 needs none.
PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/multilingual-e5-base": ("query: ", "passage: "),
    "intfloat/multilingual-e5-large": ("query: ", "passage: "),
}
MAX_SEQ_LENGTH = 512
BATCH_SIZE = 16


def model_slug(model_name: str) -> str:
    return model_name.split("/")[-1].lower().replace(".", "-")


@lru_cache(maxsize=2)
def load_model(model_name: str):  # noqa: ANN201 - SentenceTransformer
    import os

    import torch
    from sentence_transformers import SentenceTransformer

    from backend.app.config import configure_hf_cache

    # same knob as the reranker: physical cores locally, 2 to simulate HF Spaces CPU basic
    torch.set_num_threads(int(os.environ.get("SAHAYI_TORCH_THREADS", 8)))
    model = SentenceTransformer(model_name, device="cpu", cache_folder=str(configure_hf_cache()))
    model.max_seq_length = MAX_SEQ_LENGTH
    return model


class EmbeddingCache:
    """Vectors keyed by sha1(text) in data/cache/emb/{model}.npz + .json index."""

    def __init__(self, directory: Path, model_name: str) -> None:
        self.base = directory / model_slug(model_name)
        self.keys: list[str] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)
        if self.base.with_suffix(".json").exists():
            self.keys = json.loads(self.base.with_suffix(".json").read_text("utf-8"))
            self.vectors = np.load(self.base.with_suffix(".npz"))["v"]
        self.index = {k: i for i, k in enumerate(self.keys)}

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def get_many(self, texts: Sequence[str]) -> list[np.ndarray | None]:
        return [
            self.vectors[self.index[k]] if (k := self.key(t)) in self.index else None for t in texts
        ]

    def add(self, texts: Sequence[str], vectors: np.ndarray) -> None:
        new_keys = [self.key(t) for t in texts]
        self.vectors = vectors if not self.keys else np.vstack([self.vectors, vectors])
        self.keys += new_keys
        self.index = {k: i for i, k in enumerate(self.keys)}

    def save(self) -> None:
        self.base.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.base.with_suffix(".npz"), v=self.vectors)
        self.base.with_suffix(".json").write_text(json.dumps(self.keys), encoding="utf-8")


def embed_passages(
    texts: Sequence[str], model_name: str, cache: EmbeddingCache | None = None
) -> np.ndarray:
    """Normalised passage vectors, embedding only texts missing from the cache."""
    prefix = PREFIXES.get(model_name, ("", ""))[1]
    cached = cache.get_many(texts) if cache else [None] * len(texts)
    missing = [i for i, v in enumerate(cached) if v is None]
    if missing:
        model = load_model(model_name)
        new = model.encode(
            [prefix + texts[i] for i in missing],
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        if cache:
            cache.add([texts[i] for i in missing], new)
            cache.save()
        for j, i in enumerate(missing):
            cached[i] = new[j]
    return np.vstack(cached).astype(np.float32)


def embed_queries(queries: Sequence[str], model_name: str) -> np.ndarray:
    """Normalised query vectors, all variants in one forward pass."""
    prefix = PREFIXES.get(model_name, ("", ""))[0]
    vecs = load_model(model_name).encode(
        [prefix + q for q in queries], normalize_embeddings=True, batch_size=len(queries)
    )
    return np.asarray(vecs, dtype=np.float32)


class OnnxQueryEncoder:
    """int8 ONNX query encoder (deploy image: no torch). Documents stay fp32 vectors."""

    def __init__(self, onnx_file: str) -> None:
        import os

        import onnxruntime as ort
        from transformers import AutoTokenizer

        from backend.app.retrieval.rerank import ONNX_DIR

        path = ONNX_DIR / onnx_file
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(os.environ.get("SAHAYI_TORCH_THREADS", 8))
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.tok = AutoTokenizer.from_pretrained(str(path.parent))

    def encode(self, queries: Sequence[str]) -> np.ndarray:
        enc = self.tok(list(queries), padding=True, truncation=True, max_length=MAX_SEQ_LENGTH,
                       return_tensors="np")  # fmt: skip
        feed = {k: enc[k].astype("int64") for k in ("input_ids", "attention_mask")}
        return np.asarray(self.session.run(None, feed)[0], dtype=np.float32)


@lru_cache(maxsize=1)
def onnx_query_encoder(onnx_file: str) -> OnnxQueryEncoder:
    return OnnxQueryEncoder(onnx_file)


def embed_query(query: str, model_name: str) -> np.ndarray:
    return embed_queries([query], model_name)[0]


class DenseIndex:
    """A Qdrant collection of chunk vectors (cosine) with chunk_id payloads."""

    def __init__(self, qdrant_path: Path, collection: str) -> None:
        from qdrant_client import QdrantClient

        self.client = QdrantClient(path=str(qdrant_path))
        self.collection = collection

    def build(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=vectors.shape[1], distance=models.Distance.COSINE
            ),
        )
        self.client.upload_points(
            self.collection,
            points=[
                models.PointStruct(id=i, vector=vectors[i].tolist(), payload={"chunk_id": cid})
                for i, cid in enumerate(chunk_ids)
            ],
        )

    def search(self, query_vector: np.ndarray, k: int = 20) -> list[tuple[str, float]]:
        hits = self.client.query_points(
            self.collection, query=query_vector.tolist(), limit=k, with_payload=True
        ).points
        return [(h.payload["chunk_id"], float(h.score)) for h in hits]

    def close(self) -> None:
        self.client.close()
