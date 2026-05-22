from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets.rlpr_restoration_dataset import RLPRRestorationDataset
from engine.inference_pipeline import ALPRPipeline
from metrics.psnr import compute_psnr
from metrics.ssim import compute_ssim
from models.model_registry import ModelRegistry, initialize_registry
from utils.checkpoint_utils import resolve_best_checkpoint
from utils.config_loader import ConfigManager


def compute_cer(pred: str, gt: str) -> float:
    import Levenshtein

    pred = pred.lower().replace(" ", "")
    gt = gt.lower().replace(" ", "")
    if not gt:
        return 0.0
    return Levenshtein.distance(pred, gt) / len(gt)


def resolve_models(arg_models: List[str]) -> List[str]:
    initialize_registry()
    available = sorted(ModelRegistry.list_models().keys())
    if not arg_models or arg_models == ["all"]:
        return available
    invalid = [name for name in arg_models if name not in available]
    if invalid:
        raise ValueError(f"Unknown model(s): {invalid}. Available: {available}")
    return arg_models


def input_mode_for_model(model_name: str) -> str:
    initialize_registry()
    meta = ModelRegistry.get_metadata(model_name)
    return "sequence" if int(meta.input_channels) == 15 else "center_upscale"


def load_eval_dataset(config_path: str, model_name: str, split: str):
    cfg = ConfigManager(".").loader.load(Path(config_path).name)
    local_cfg = copy.deepcopy(cfg)
    local_cfg.training["input_mode"] = input_mode_for_model(model_name)
    return RLPRRestorationDataset.from_config(
        local_cfg,
        project_root=".",
        split=split,
        augmentation_enabled=False,
    )


def evaluate_pipeline(
    pipeline: ALPRPipeline,
    dataset: RLPRRestorationDataset,
    max_samples: int,
) -> Dict[str, float]:
    sample_count = min(max_samples, len(dataset)) if max_samples > 0 else len(dataset)

    psnr_values: List[float] = []
    ssim_values: List[float] = []
    cer_values: List[float] = []
    seq_hits = 0
    conf_values: List[float] = []
    latency_ms: List[float] = []

    for idx in range(sample_count):
        sample = dataset[idx]
        inp = sample["input"].unsqueeze(0).to(pipeline.device)
        target = sample["target"]
        gt_text = sample["plate_text_compact"]

        start = time.perf_counter()
        out = pipeline(inp)
        latency_ms.append((time.perf_counter() - start) * 1000.0)

        refined = out["refined"][0].detach().cpu()
        pred_text = out["text"][0] if out["text"] else ""
        confidence = float(out["confidence"][0]) if out["confidence"] else 0.0

        psnr_values.append(compute_psnr(refined, target))
        ssim_values.append(compute_ssim(refined, target))
        cer = compute_cer(pred_text, gt_text)
        cer_values.append(cer)
        seq_hits += int(pred_text.lower().replace(" ", "") == gt_text.lower().replace(" ", ""))
        conf_values.append(confidence)

    denom = max(sample_count, 1)
    return {
        "samples": sample_count,
        "psnr": sum(psnr_values) / denom,
        "ssim": sum(ssim_values) / denom,
        "cer": sum(cer_values) / denom,
        "seq_acc": seq_hits / denom,
        "confidence": sum(conf_values) / denom,
        "latency_ms": sum(latency_ms) / denom,
        "fps": 1000.0 / max(sum(latency_ms) / denom, 1e-6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark restoration models with modular OCR backends.")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names to evaluate or 'all'.")
    parser.add_argument("--ocr-backends", nargs="+", default=["fast_plate_ocr", "parseq"], help="OCR backends to compare.")
    parser.add_argument("--config", type=str, default="configs/train_restoration.yaml", help="Merged training config entry.")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--max-samples", type=int, default=20, help="Max dataset samples per model/backend pair. Use <=0 for full split.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prefer-metric", type=str, default="psnr", choices=["psnr", "cer"], help="Checkpoint selection heuristic.")
    parser.add_argument("--output", type=str, default="outputs/benchmarks/alpr_model_comparison.csv")
    args = parser.parse_args()

    models = resolve_models(args.models)
    rows = []

    for model_name in models:
        dataset = load_eval_dataset(args.config, model_name, args.split)
        ckpt = resolve_best_checkpoint(model_name, prefer_metric=args.prefer_metric)
        for ocr_backend in args.ocr_backends:
            pipeline = ALPRPipeline(
                model_name=model_name,
                checkpoint_path=str(ckpt),
                ocr_backend=ocr_backend,
                device=args.device,
            )
            metrics = evaluate_pipeline(pipeline, dataset, args.max_samples)
            rows.append(
                {
                    "model": model_name,
                    "checkpoint": str(ckpt),
                    "ocr_backend": ocr_backend,
                    **metrics,
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values(["seq_acc", "cer", "psnr"], ascending=[False, True, False]).reset_index(drop=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nALPR model comparison")
    print(df.to_string(index=False))
    print(f"\nSaved benchmark table to {output_path}")


if __name__ == "__main__":
    main()
