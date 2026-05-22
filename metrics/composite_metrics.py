"""
Composite metrics module for restoration quality evaluation.

Provides unified wrappers for:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index Measure)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Runtime profiling (FPS, latency, VRAM usage)
- Model complexity (FLOPs, parameter count)
"""

import time
import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import psutil
import os

from metrics.psnr import compute_psnr
from metrics.ssim import compute_ssim


class CompositeMetricComputer:
    """Unified interface for computing restoration quality metrics."""

    def __init__(self, lpips_enabled: bool = True, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize metric computers.

        Args:
            lpips_enabled: Whether to compute LPIPS (requires lpips package)
            device: Compute device ('cuda' or 'cpu')
        """
        self.device = device
        self.lpips_enabled = lpips_enabled
        self.lpips_fn = None

        if lpips_enabled:
            try:
                import lpips as lpips_module
                self.lpips_fn = lpips_module.LPIPS(net='alex', verbose=False).to(device)
                self.lpips_fn.eval()
                for param in self.lpips_fn.parameters():
                    param.requires_grad = False
            except ImportError:
                print("Warning: lpips not installed. LPIPS computation will be disabled.")
                self.lpips_enabled = False

    def compute_psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute PSNR between prediction and target.

        Args:
            pred: Predicted image (B, C, H, W) in [0, 1]
            target: Target image (B, C, H, W) in [0, 1]

        Returns:
            PSNR value in dB
        """
        return compute_psnr(pred, target)

    def compute_ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute SSIM between prediction and target.

        Args:
            pred: Predicted image (B, C, H, W) in [0, 1]
            target: Target image (B, C, H, W) in [0, 1]

        Returns:
            SSIM value in [0, 1]
        """
        return compute_ssim(pred, target)

    def compute_lpips(self, pred: torch.Tensor, target: torch.Tensor, min_size: int = 32) -> Optional[float]:
        """
        Compute LPIPS (perceptual loss) between prediction and target.

        Args:
            pred: Predicted image (B, C, H, W) in [0, 1]
            target: Target image (B, C, H, W) in [0, 1]
            min_size: Minimum spatial size to compute LPIPS (smaller images resized)

        Returns:
            LPIPS value or None if LPIPS disabled
        """
        if not self.lpips_enabled:
            return None

        h, w = pred.shape[-2], pred.shape[-1]
        if h < min_size or w < min_size:
            new_h = max(h, min_size)
            new_w = max(w, min_size)
            is_3d = pred.ndim == 3
            if is_3d:
                pred = pred.unsqueeze(0)
                target = target.unsqueeze(0)
            pred = F.interpolate(pred, size=(new_h, new_w), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(new_h, new_w), mode='bilinear', align_corners=False)
            if is_3d:
                pred = pred.squeeze(0)
                target = target.squeeze(0)

        with torch.no_grad():
            # LPIPS expects values in [-1, 1]
            pred_normalized = 2 * pred - 1
            target_normalized = 2 * target - 1
            lpips_val = self.lpips_fn(pred_normalized, target_normalized)
        return lpips_val.item()

    def compute_all_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        compute_lpips: bool = True,
    ) -> Dict[str, Optional[float]]:
        """
        Compute all available metrics at once.

        Args:
            pred: Predicted image (B, C, H, W) in [0, 1]
            target: Target image (B, C, H, W) in [0, 1]
            compute_lpips: Whether to compute LPIPS

        Returns:
            Dictionary with keys: psnr, ssim, lpips (if enabled and computed)
        """
        metrics = {}
        metrics["psnr"] = self.compute_psnr(pred, target)
        metrics["ssim"] = self.compute_ssim(pred, target)
        if compute_lpips:
            metrics["lpips"] = self.compute_lpips(pred, target)
        return metrics


class RuntimeProfiler:
    """Profiles model inference runtime characteristics."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize runtime profiler.

        Args:
            model: PyTorch model to profile
            device: Compute device
        """
        self.model = model
        self.device = device
        self.process = psutil.Process(os.getpid())

    def profile_inference(
        self,
        input_tensor: torch.Tensor,
        num_runs: int = 10,
        warmup_runs: int = 3,
    ) -> Dict[str, float]:
        """
        Profile model inference time and memory usage.

        Args:
            input_tensor: Model input (B, C, H, W)
            num_runs: Number of inference runs to average
            warmup_runs: Number of warmup runs before timing

        Returns:
            Dictionary with:
                - latency_ms: Average inference time (ms)
                - fps: Frames per second
                - vram_mb: GPU memory used (MB)
                - ram_mb: CPU memory used (MB)
        """
        self.model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = self.model(input_tensor.to(self.device))

        # Measure GPU memory before
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        ram_before = self.process.memory_info().rss / (1024 ** 2)

        # Timing
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start = time.time()
                _ = self.model(input_tensor.to(self.device))
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                times.append(time.time() - start)

        avg_time = sum(times) / len(times)
        latency_ms = avg_time * 1000
        fps = 1.0 / avg_time

        vram_mb = 0.0
        if torch.cuda.is_available():
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        ram_after = self.process.memory_info().rss / (1024 ** 2)
        ram_mb = ram_after - ram_before

        return {
            "latency_ms": latency_ms,
            "fps": fps,
            "vram_mb": vram_mb,
            "ram_mb": ram_mb,
        }


class ModelComplexity:
    """Computes model complexity metrics (FLOPs, parameters)."""

    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def estimate_flops(model: torch.nn.Module, input_shape: Tuple[int, ...]) -> Optional[int]:
        """
        Estimate FLOPs for a forward pass.

        Uses fvcore.nn.FlopCountAnalysis if available, otherwise returns None.

        Args:
            model: PyTorch model
            input_shape: Input tensor shape (e.g., (1, 3, 256, 256))

        Returns:
            Estimated FLOPs or None if fvcore not available
        """
        try:
            from fvcore.nn import FlopCountAnalysis
            dummy_input = torch.randn(input_shape)
            flops = FlopCountAnalysis(model, dummy_input).total()
            return int(flops)
        except ImportError:
            print("Warning: fvcore not installed. FLOPs estimation requires: pip install fvcore")
            return None

    @staticmethod
    def format_complexity(num_params: int, flops: Optional[int] = None) -> str:
        """
        Format complexity metrics for display.

        Args:
            num_params: Parameter count
            flops: FLOPs or None

        Returns:
            Formatted string
        """
        param_str = f"{num_params / 1e6:.2f}M" if num_params >= 1e6 else f"{num_params / 1e3:.1f}K"
        result = f"Params: {param_str}"

        if flops is not None:
            flop_str = f"{flops / 1e9:.2f}G" if flops >= 1e9 else f"{flops / 1e6:.2f}M"
            result += f" | FLOPs: {flop_str}"

        return result


class CompositeMetricsLogger:
    """Helper for logging all metrics to MLflow and local storage."""

    def __init__(self, experiment_name: str, run_name: str, use_mlflow: bool = True):
        """
        Initialize metrics logger.

        Args:
            experiment_name: MLflow experiment name
            run_name: MLflow run name
            use_mlflow: Whether to log to MLflow
        """
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.use_mlflow = use_mlflow
        self.metrics_history = []

        if use_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
                mlflow.set_experiment(experiment_name)
            except ImportError:
                print("Warning: mlflow not installed. Logging disabled.")
                self.use_mlflow = False

    def log_metrics(self, metrics: Dict[str, float], step: int = None, prefix: str = ""):
        """
        Log metrics to MLflow and memory.

        Args:
            metrics: Dictionary of metric name -> value
            step: Training step/epoch
            prefix: Prefix for metric names (e.g., "val/")
        """
        self.metrics_history.append((step, prefix, metrics.copy()))

        if self.use_mlflow:
            for key, val in metrics.items():
                if val is not None:
                    metric_name = f"{prefix}{key}" if prefix else key
                    if step is not None:
                        self.mlflow.log_metric(metric_name, val, step=step)
                    else:
                        self.mlflow.log_metric(metric_name, val)

    def get_metrics_summary(self) -> Dict:
        """Get summary of logged metrics."""
        summary = {}
        for step, prefix, metrics in self.metrics_history:
            for key, val in metrics.items():
                metric_name = f"{prefix}{key}" if prefix else key
                if metric_name not in summary:
                    summary[metric_name] = []
                summary[metric_name].append(val)
        return summary


if __name__ == "__main__":
    # Test metrics
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy tensors
    pred = torch.rand(1, 3, 256, 256).to(device)
    target = torch.rand(1, 3, 256, 256).to(device)
    
    # Test CompositeMetricComputer
    metric_computer = CompositeMetricComputer(device=device)
    metrics = metric_computer.compute_all_metrics(pred, target)
    print("Metrics:", metrics)
    
    # Test ModelComplexity
    from models.restoration.swinir_unet import SwinIRUNetHybrid
    model = SwinIRUNetHybrid().to(device)
    params = ModelComplexity.count_parameters(model)
    flops = ModelComplexity.estimate_flops(model, (1, 3, 256, 256))
    complexity = ModelComplexity.format_complexity(params, flops)
    print("Complexity:", complexity)
    
    # Test RuntimeProfiler
    profiler = RuntimeProfiler(model, device=device)
    runtime_metrics = profiler.profile_inference(torch.rand(1, 3, 256, 256), num_runs=5)
    print("Runtime metrics:", runtime_metrics)
