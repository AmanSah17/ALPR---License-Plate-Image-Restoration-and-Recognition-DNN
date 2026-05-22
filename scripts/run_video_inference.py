from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path
from typing import Deque, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.inference_pipeline import ALPRPipeline
from models.model_registry import ModelRegistry, initialize_registry
from utils.checkpoint_utils import resolve_best_checkpoint


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def parse_roi(roi_text: str | None):
    if not roi_text:
        return None
    parts = [int(p.strip()) for p in roi_text.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be 'x,y,w,h'")
    return tuple(parts)


def model_uses_sequence(model_name: str) -> bool:
    initialize_registry()
    meta = ModelRegistry.get_metadata(model_name)
    return int(meta.input_channels) == 15


def bgr_to_chw_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).float().permute(2, 0, 1) / 255.0
    return tensor.contiguous()


def tensor_to_bgr_image(image: torch.Tensor, size: Tuple[int, int]) -> np.ndarray:
    arr = image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    arr = (arr * 255.0).astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return cv2.resize(arr, size, interpolation=cv2.INTER_CUBIC)


def build_model_input(crop_bgr: np.ndarray, use_sequence: bool, buffer: Deque[torch.Tensor]) -> torch.Tensor:
    frame_tensor = bgr_to_chw_tensor(crop_bgr)
    if not use_sequence:
        return frame_tensor.unsqueeze(0)

    buffer.append(frame_tensor)
    while len(buffer) < 5:
        buffer.appendleft(frame_tensor.clone())
    sequence = list(buffer)[-5:]
    stacked = torch.cat(sequence, dim=0)
    return stacked.unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live ALPR inference on a webcam/video ROI. Input is assumed to already contain a plate crop or fixed plate ROI."
    )
    parser.add_argument("--model", type=str, default="swinir_base")
    parser.add_argument("--ckpt", type=str, default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--ocr-backend", type=str, default="fast_plate_ocr")
    parser.add_argument("--source", type=str, default="0", help="Camera index or video path.")
    parser.add_argument("--roi", type=str, default=None, help="Fixed ROI as x,y,w,h. If omitted, the full frame is used.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the annotated video.")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional frame cap for non-interactive runs.")
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt) if args.ckpt else resolve_best_checkpoint(args.model, prefer_metric="psnr")
    pipeline = ALPRPipeline(
        model_name=args.model,
        checkpoint_path=str(ckpt_path),
        ocr_backend=args.ocr_backend,
        device=args.device,
    )

    source = parse_source(args.source)
    roi = parse_roi(args.roi)
    use_sequence = model_uses_sequence(args.model)
    frame_buffer: Deque[torch.Tensor] = collections.deque(maxlen=5)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    writer = None
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    frame_idx = 0
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            display = frame.copy()
            h, w = frame.shape[:2]
            x, y, rw, rh = roi if roi is not None else (0, 0, w, h)
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))

            crop = frame[y:y + rh, x:x + rw]
            model_input = build_model_input(crop, use_sequence, frame_buffer)

            start = time.perf_counter()
            out = pipeline(model_input)
            elapsed = time.perf_counter() - start
            instant_fps = 1.0 / max(elapsed, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0.0 else (0.9 * smoothed_fps + 0.1 * instant_fps)

            refined = tensor_to_bgr_image(out["refined"][0], (rw, rh))
            pred_text = out["text"][0] if out["text"] else ""
            confidence = float(out["confidence"][0]) if out["confidence"] else 0.0

            display[y:y + rh, x:x + rw] = refined
            cv2.rectangle(display, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
            cv2.putText(display, f"{args.model} | {args.ocr_backend}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"Plate: {pred_text}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(display, f"Conf: {confidence:.2f} | FPS: {smoothed_fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.imshow("ALPR Video Inference", display)
            if writer is not None:
                writer.write(display)

            frame_idx += 1
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
