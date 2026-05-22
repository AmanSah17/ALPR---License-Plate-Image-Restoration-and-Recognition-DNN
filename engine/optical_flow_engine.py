"""
Optical flow inference engine: RAFT -> temporal filter -> warp -> save artifacts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from logging_utils.phase_outputs import PhaseOutputPaths
from models.optical_flow.flow_utils import FlowUtils, flow_to_numpy, numpy_hwc_to_chw_float
from models.optical_flow.frame_warping import FrameWarper
from models.optical_flow.raft_wrapper import RAFTSmallWrapper
from models.optical_flow.temporal_filter import TemporalFlowFilter
from utils.gpu_utils import GPUManager
from utils.image_utils import save_image_rgb
from utils.profiler import RuntimeProfiler
from utils.timing import BenchmarkTimer

logger = logging.getLogger(__name__)


@dataclass
class OpticalFlowResult:
    """Outputs from a multi-frame optical flow pass."""

    sample_id: str
    reference_index: int
    flows: torch.Tensor  # (T, 2, H, W) filtered, toward reference
    warped: torch.Tensor  # (T, 3, H, W)
    frame_indices: List[int]
    metrics: Dict[str, float] = field(default_factory=dict)
    artifact_paths: Dict[str, str] = field(default_factory=dict)


class OpticalFlowEngine:
    """
    End-to-end optical flow pipeline for RLPR sequences.

    Steps:
        1. Select reference frame (center by default).
        2. Estimate flow from each frame -> reference with RAFT-small.
        3. Temporally filter flow stack.
        4. Warp frames to reference coordinates.
        5. Save flows, warped frames, overlays, error maps to phase folders.
    """

    def __init__(
        self,
        raft: RAFTSmallWrapper,
        temporal_filter: TemporalFlowFilter,
        warper: FrameWarper,
        output_paths: PhaseOutputPaths,
        gpu_manager: Optional[GPUManager] = None,
        save_flow_numpy: bool = True,
        save_warped_png: bool = True,
        save_overlays: bool = True,
        save_error_maps: bool = True,
        max_frame_pairs: Optional[int] = None,
    ) -> None:
        self.raft = raft
        self.filter = temporal_filter
        self.warper = warper
        self.outputs = output_paths
        self.gpu = gpu_manager or GPUManager()
        self.save_flow_numpy = save_flow_numpy
        self.save_warped_png = save_warped_png
        self.save_overlays = save_overlays
        self.save_error_maps = save_error_maps
        self.max_frame_pairs = max_frame_pairs

    def _select_frame_indices(self, num_frames: int, reference_index: int) -> List[int]:
        """Indices to process (all except reference, optionally truncated)."""
        indices = [i for i in range(num_frames) if i != reference_index]
        if self.max_frame_pairs is not None:
            # Keep frames closest to reference first
            indices.sort(key=lambda i: abs(i - reference_index))
            indices = indices[: self.max_frame_pairs]
        return indices

    def run(
        self,
        frames_hwc: np.ndarray,
        sample_id: str,
        reference_index: int,
    ) -> OpticalFlowResult:
        """
        Run pipeline on uint8 ``(T, H, W, 3)`` sequence.

        Args:
            frames_hwc: Raw RGB frames.
            sample_id: Sample identifier for filenames.
            reference_index: Zero-based reference (15 for frame 16.png).

        Returns:
            ``OpticalFlowResult`` with tensors and saved artifact paths.
        """
        t, h, w, _ = frames_hwc.shape
        tensors = [numpy_hwc_to_chw_float(frames_hwc[i]) for i in range(t)]
        ref = tensors[reference_index]
        indices = self._select_frame_indices(t, reference_index)

        logger.info(
            "Optical flow [%s]: %d frames, reference=%d, processing %d pairs",
            sample_id,
            t,
            reference_index,
            len(indices),
        )

        flows_list: List[torch.Tensor] = []
        warped_list: List[torch.Tensor] = []
        processed_indices: List[int] = []

        timer = BenchmarkTimer(name="flow_pair", warmup=0)
        for idx in indices:
            with timer.measure():
                # Flow from frame idx -> reference
                flow, _ = self.raft.predict_pair(tensors[idx], ref)
            flows_list.append(flow)
            warped = self.warper.warp(tensors[idx], flow)
            warped_list.append(warped)
            processed_indices.append(idx)

        if not flows_list:
            raise RuntimeError("No frame pairs processed for optical flow.")

        flows_stack = torch.stack(flows_list, dim=0)
        flows_filtered = self.filter.filter(flows_stack)
        warped_stack = self.warper.warp_batch(
            torch.stack([tensors[i] for i in processed_indices], dim=0),
            flows_filtered,
        )

        # Metrics
        mag_mean = float(FlowUtils.flow_magnitude(flows_filtered).mean().item())
        consistency = FlowUtils.temporal_consistency(flows_filtered)
        warp_err = float(
            torch.mean(
                torch.abs(
                    ref.unsqueeze(0).expand_as(warped_stack) - warped_stack
                )
            ).item()
        )
        timing = timer.stats()

        metrics = {
            "flow_magnitude_mean": mag_mean,
            "temporal_consistency": consistency,
            "warp_l1_error_mean": warp_err,
            "num_pairs": float(len(indices)),
            "inference_mean_ms": timing.mean_ms,
            "inference_fps": timing.fps,
        }

        artifact_paths = self._save_artifacts(
            sample_id,
            reference_index,
            processed_indices,
            flows_filtered,
            warped_stack,
            ref,
            tensors,
        )

        return OpticalFlowResult(
            sample_id=sample_id,
            reference_index=reference_index,
            flows=flows_filtered,
            warped=warped_stack,
            frame_indices=processed_indices,
            metrics=metrics,
            artifact_paths=artifact_paths,
        )

    def _save_artifacts(
        self,
        sample_id: str,
        ref_idx: int,
        frame_indices: List[int],
        flows: torch.Tensor,
        warped: torch.Tensor,
        reference: torch.Tensor,
        originals: List[torch.Tensor],
    ) -> Dict[str, str]:
        """Persist numpy/png artifacts into phase output folders."""
        paths: Dict[str, str] = {}
        sample_dir_name = sample_id

        for i, fidx in enumerate(frame_indices):
            tag = f"{sample_dir_name}_frame{fidx+1:02d}_to_ref{ref_idx+1:02d}"

            if self.save_flow_numpy:
                flow_path = self.outputs.flows / f"{tag}_flow.npy"
                np.save(flow_path, flow_to_numpy(flows[i]))
                paths[f"flow_{fidx}"] = str(flow_path)

            if self.save_warped_png:
                warped_np = (
                    warped[i].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255
                ).astype(np.uint8)
                warped_path = self.outputs.warped / f"{tag}_warped.png"
                save_image_rgb(warped_np, warped_path)
                paths[f"warped_{fidx}"] = str(warped_path)

            if self.save_error_maps:
                err = self.warper.warp_error_map(reference, warped[i])
                err_np = (err[0].cpu().numpy() * 255).astype(np.uint8)
                err_path = self.outputs.error_maps / f"{tag}_error.png"
                save_image_rgb(err_np, err_path)
                paths[f"error_{fidx}"] = str(err_path)

            if self.save_overlays:
                ref_np = (
                    reference.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255
                ).astype(np.uint8)
                src_np = (
                    originals[fidx].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255
                ).astype(np.uint8)
                blend = (0.5 * ref_np.astype(np.float32) + 0.5 * warped_np.astype(np.float32))
                overlay_path = self.outputs.overlays / f"{tag}_overlay.png"
                save_image_rgb(blend.astype(np.uint8), overlay_path)
                paths[f"overlay_{fidx}"] = str(overlay_path)

        metrics_path = self.outputs.metrics / f"{sample_dir_name}_flow_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "sample_id": sample_id,
                    "reference_index": ref_idx,
                    "frame_indices": frame_indices,
                },
                handle,
                indent=2,
            )
        paths["metrics_json"] = str(metrics_path)
        return paths

    def benchmark_pair_inference(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor,
        warmup: int = 3,
        iterations: int = 10,
    ) -> Dict[str, float]:
        """
        Benchmark RAFT pair inference (for MLflow reporting).

        Returns:
            Timing and GPU memory dict.
        """
        profiler = RuntimeProfiler(name="raft_pair", gpu_manager=self.gpu, warmup=warmup)
        profiler.begin_run()
        timer = profiler.timer.stage("raft_forward")

        def _run() -> None:
            self.raft.predict_pair(image1, image2)

        from utils.profiler import run_timed_iterations

        stats = run_timed_iterations(_run, timer, iterations, warmup)
        report = profiler.end_run()
        return {
            **stats.to_dict(),
            **report.memory,
        }

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        output_paths: PhaseOutputPaths,
        gpu_manager: Optional[GPUManager] = None,
    ) -> "OpticalFlowEngine":
        """Factory from merged optical_flow config."""
        of = cfg.optical_flow if hasattr(cfg, "optical_flow") else cfg
        out = cfg.output if hasattr(cfg, "output") else {}
        hw = cfg.hardware if hasattr(cfg, "hardware") else cfg
        gpu = gpu_manager or GPUManager.from_config(hw)
        max_fp = of.get("max_frame_pairs") if hasattr(of, "get") else None
        return cls(
            raft=RAFTSmallWrapper.from_config(cfg, gpu_manager=gpu),
            temporal_filter=TemporalFlowFilter.from_config(of),
            warper=FrameWarper(),
            output_paths=output_paths,
            gpu_manager=gpu,
            save_flow_numpy=bool(out.get("save_flow_numpy", True)),
            save_warped_png=bool(out.get("save_warped_png", True)),
            save_overlays=bool(out.get("save_overlays", True)),
            save_error_maps=bool(out.get("save_error_maps", True)),
            max_frame_pairs=int(max_fp) if max_fp is not None else None,
        )
