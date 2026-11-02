"""Export a cross-encoder to ONNX and quantise it to int8 (dynamic, weights only).

    python -m backend.app.retrieval.export_onnx --model BAAI/bge-reranker-v2-m3

Writes data/cache/onnx/{model-slug}/model.onnx (fp32, external weights) and
model_int8.onnx. Uses torch.onnx + onnxruntime.quantization only (optimum pins an
older transformers and breaks sentence-transformers).
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ONNX_DIR = ROOT / "data" / "cache" / "onnx"


def onnx_dir(model_name: str) -> Path:
    return ONNX_DIR / model_name.split("/")[-1]


def export(model_name: str, opset: int = 17) -> Path:
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from backend.app.config import configure_hf_cache

    cache = configure_hf_cache()
    out = onnx_dir(model_name)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
    tok.save_pretrained(out)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache)
    model.eval()
    sample = tok(["query"], ["passage text"], return_tensors="pt")
    fp32 = out / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "logits": {0: "batch"},
            },
            opset_version=opset,
            dynamo=False,
        )
    int8 = out / "model_int8.onnx"
    del model  # free ~2 GB before quantisation loads the graph
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"exported {fp32} and {int8} ({int8.stat().st_size / 1e6:.0f} MB)")
    return int8


def export_query_encoder(model_name: str = "BAAI/bge-m3", opset: int = 17) -> Path:
    """Dense query encoder (normalised CLS vector, as sentence-transformers gives for bge-m3)."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoTokenizer

    from backend.app.config import configure_hf_cache

    class Encoder(torch.nn.Module):
        def __init__(self, model: torch.nn.Module) -> None:
            super().__init__()
            self.model = model

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            cls = self.model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0]
            return torch.nn.functional.normalize(cls, dim=1)

    cache = configure_hf_cache()
    out = ONNX_DIR / f"{model_name.split('/')[-1]}-query"
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_name, cache_dir=cache)
    tok.save_pretrained(out)
    model = Encoder(AutoModel.from_pretrained(model_name, cache_dir=cache)).eval()
    sample = tok(["a query"], return_tensors="pt")
    fp32 = out / "model.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (sample["input_ids"], sample["attention_mask"]),
            str(fp32),
            input_names=["input_ids", "attention_mask"],
            output_names=["embedding"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "embedding": {0: "batch"},
            },
            opset_version=opset,
            dynamo=False,
        )
    int8 = out / "model_int8.onnx"
    del model
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    print(f"exported {int8} ({int8.stat().st_size / 1e6:.0f} MB)")
    return int8


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--query-encoder", action="store_true", help="bge-m3 dense query encoder")
    args = ap.parse_args()
    if args.query_encoder:
        export_query_encoder()
    else:
        export(args.model)


if __name__ == "__main__":
    main()
