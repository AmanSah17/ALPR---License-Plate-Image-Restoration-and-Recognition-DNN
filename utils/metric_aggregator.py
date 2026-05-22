"""
Metrics aggregation and export module.

Handles:
- Collection of metrics across training epochs
- CSV export for cross-model comparison
- MLflow logging integration
- Summary report generation
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import statistics


class MetricAggregator:
    """Aggregates and exports metrics from training runs."""

    def __init__(self, experiment_name: str, model_name: str, output_dir: Path = None):
        """
        Initialize metric aggregator.

        Args:
            experiment_name: Name of experiment (for logging)
            model_name: Name of model variant being trained
            output_dir: Directory to save CSV exports (default: outputs/metrics/)
        """
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/metrics")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_history: List[Dict[str, Any]] = []
        self.training_config: Dict[str, Any] = {}

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        learning_rate: Optional[float] = None,
    ):
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
            learning_rate: Current learning rate (optional)
        """
        entry = {"epoch": epoch}

        # Add training metrics
        for key, val in train_metrics.items():
            if val is not None:
                entry[f"train/{key}"] = val

        # Add validation metrics
        for key, val in val_metrics.items():
            if val is not None:
                entry[f"val/{key}"] = val

        # Add learning rate
        if learning_rate is not None:
            entry["learning_rate"] = learning_rate

        self.metrics_history.append(entry)

    def log_config(self, config: Dict[str, Any]):
        """
        Store training configuration.

        Args:
            config: Dictionary with training hyperparameters
        """
        self.training_config = config.copy()

    def get_best_metrics(self, metric_name: str = "val/psnr") -> Dict[str, Any]:
        """
        Get best epoch for a specific metric.

        Args:
            metric_name: Metric to optimize for (e.g., 'val/psnr')

        Returns:
            Dictionary with best epoch data and value
        """
        if not self.metrics_history:
            return {}

        valid_entries = [e for e in self.metrics_history if metric_name in e]
        if not valid_entries:
            return {}

        best_entry = max(valid_entries, key=lambda x: x[metric_name])
        return best_entry

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics for all metrics.

        Returns:
            Dictionary with min/max/mean/std for each metric
        """
        if not self.metrics_history:
            return {}

        summary = {}

        # Get all metric keys
        all_keys = set()
        for entry in self.metrics_history:
            all_keys.update(entry.keys())

        # Compute statistics for each metric (skip epoch and lr)
        for key in all_keys:
            if key in ("epoch", "learning_rate"):
                continue

            values = [e[key] for e in self.metrics_history if key in e and isinstance(e[key], (int, float))]
            if values:
                summary[key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                }

        return summary

    def export_to_csv(self, filename: Optional[str] = None) -> Path:
        """
        Export metrics to CSV file.

        Args:
            filename: Output filename (default: {model_name}_metrics.csv)

        Returns:
            Path to exported CSV file
        """
        if not self.metrics_history:
            print("Warning: No metrics to export.")
            return None

        if filename is None:
            filename = f"{self.model_name}_metrics.csv"

        output_path = self.output_dir / filename

        # Get all unique keys
        all_keys = set()
        for entry in self.metrics_history:
            all_keys.update(entry.keys())

        # Sort keys: epoch first, then alphabetically
        all_keys = sorted(all_keys)
        all_keys.remove("epoch")
        all_keys = ["epoch"] + all_keys

        # Write CSV
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()

            for entry in self.metrics_history:
                row = {key: entry.get(key, "") for key in all_keys}
                writer.writerow(row)

        print(f"✓ Metrics exported to {output_path}")
        return output_path

    def export_summary_json(self, filename: Optional[str] = None) -> Path:
        """
        Export summary statistics to JSON.

        Args:
            filename: Output filename (default: {model_name}_summary.json)

        Returns:
            Path to exported JSON file
        """
        if filename is None:
            filename = f"{self.model_name}_summary.json"

        output_path = self.output_dir / filename

        summary = {
            "experiment": self.experiment_name,
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "total_epochs": len(self.metrics_history),
            "config": self.training_config,
            "metrics_summary": self.get_metrics_summary(),
            "best_metrics": self.get_best_metrics(),
        }

        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"✓ Summary exported to {output_path}")
        return output_path

    def generate_html_report(self, filename: Optional[str] = None) -> Path:
        """
        Generate an HTML report with metrics and plots.

        Args:
            filename: Output filename (default: {model_name}_report.html)

        Returns:
            Path to exported HTML file
        """
        if filename is None:
            filename = f"{self.model_name}_report.html"

        output_path = self.output_dir / filename

        summary = self.get_metrics_summary()
        best = self.get_best_metrics()

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Training Report - {self.model_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        h2 {{ color: #555; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
        th {{ background-color: #007bff; color: white; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metric-name {{ font-weight: bold; color: #007bff; }}
        .best-row {{ background-color: #e7f3ff; }}
        .info-box {{ background-color: #f0f8ff; border-left: 4px solid #007bff; padding: 15px; margin: 20px 0; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #007bff; }}
        .metric-card h3 {{ margin: 0 0 10px 0; color: #007bff; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .metric-unit {{ font-size: 12px; color: #999; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Training Report: {self.model_name}</h1>
        
        <div class="info-box">
            <strong>Experiment:</strong> {self.experiment_name}<br>
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Total Epochs:</strong> {len(self.metrics_history)}<br>
        </div>

        <h2>Training Configuration</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
"""
        for key, val in self.training_config.items():
            html_content += f"<tr><td>{key}</td><td>{val}</td></tr>\n"

        html_content += """
        </table>

        <h2>Best Metrics (Optimization Target: val/psnr)</h2>
        <div class="metrics-grid">
"""
        for key, val in sorted(best.items()):
            if key not in ("epoch", "experiment", "model", "timestamp"):
                html_content += f"""
            <div class="metric-card">
                <h3>{key}</h3>
                <div class="metric-value">{val:.3f}</div>
            </div>
"""

        html_content += """
        </div>

        <h2>Metrics Summary Statistics</h2>
        <table>
            <tr><th>Metric</th><th>Min</th><th>Max</th><th>Mean</th><th>Std</th></tr>
"""
        for metric, stats in sorted(summary.items()):
            html_content += f"""
            <tr class="best-row">
                <td class="metric-name">{metric}</td>
                <td>{stats['min']:.4f}</td>
                <td>{stats['max']:.4f}</td>
                <td>{stats['mean']:.4f}</td>
                <td>{stats['std']:.4f}</td>
            </tr>
"""

        html_content += """
        </table>

        <h2>Training Progress (Last 5 Epochs)</h2>
        <table>
            <tr>
"""
        # Add headers
        if self.metrics_history:
            all_keys = sorted(self.metrics_history[-1].keys())
            for key in all_keys:
                html_content += f"<th>{key}</th>"

        html_content += """
            </tr>
"""
        # Add last 5 epochs
        for entry in self.metrics_history[-5:]:
            html_content += "<tr>"
            for key in sorted(entry.keys()):
                val = entry[key]
                if isinstance(val, float):
                    html_content += f"<td>{val:.4f}</td>"
                else:
                    html_content += f"<td>{val}</td>"
            html_content += "</tr>\n"

        html_content += """
        </table>

        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #999; font-size: 12px;">
            <p>Generated by ALPR_Research training pipeline</p>
        </footer>
    </div>
</body>
</html>
"""

        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"✓ HTML report generated: {output_path}")
        return output_path

    def export_all(self, base_filename: Optional[str] = None) -> Dict[str, Path]:
        """
        Export metrics in all formats (CSV, JSON, HTML).

        Args:
            base_filename: Base filename (without extension)

        Returns:
            Dictionary mapping format -> file path
        """
        if base_filename is None:
            base_filename = f"{self.model_name}_metrics"

        results = {}
        results["csv"] = self.export_to_csv(f"{base_filename}.csv")
        results["json"] = self.export_summary_json(f"{base_filename}_summary.json")
        results["html"] = self.generate_html_report(f"{base_filename}_report.html")

        return results


class MLflowMetricsLogger:
    """Helper for logging metrics to MLflow."""

    def __init__(self, experiment_name: str, run_name: str):
        """
        Initialize MLflow logger.

        Args:
            experiment_name: MLflow experiment name
            run_name: MLflow run name
        """
        try:
            import mlflow
            self.mlflow = mlflow
            self.mlflow.set_experiment(experiment_name)
            self.active_run = None
        except ImportError:
            print("Warning: mlflow not installed. MLflow logging disabled.")
            self.mlflow = None
            self.active_run = None

    def start_run(self, run_name: str, params: Dict[str, Any] = None):
        """Start a new MLflow run."""
        if self.mlflow is None:
            return

        self.active_run = self.mlflow.start_run(run_name=run_name)
        if params:
            self.mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to active run."""
        if self.mlflow is None or self.active_run is None:
            return

        for key, val in metrics.items():
            if val is not None:
                if step is not None:
                    self.mlflow.log_metric(key, val, step=step)
                else:
                    self.mlflow.log_metric(key, val)

    def end_run(self):
        """End active run."""
        if self.mlflow is None:
            return
        self.mlflow.end_run()


if __name__ == "__main__":
    # Example usage
    agg = MetricAggregator(experiment_name="test_exp", model_name="swinir_base")

    # Simulate training
    for epoch in range(1, 11):
        train_metrics = {"loss": 0.1 / (epoch + 1), "psnr": 15 + epoch}
        val_metrics = {"loss": 0.15 / (epoch + 1), "psnr": 14 + epoch * 0.8, "ssim": 0.4 + epoch * 0.05}
        agg.log_epoch_metrics(epoch, train_metrics, val_metrics, learning_rate=2e-4 * (0.9 ** epoch))

    # Export
    agg.log_config({"batch_size": 1, "lr": 2e-4, "optimizer": "AdamW"})
    print(f"Best metrics: {agg.get_best_metrics()}")
    print(f"Summary: {agg.get_metrics_summary()}")

    agg.export_all()
