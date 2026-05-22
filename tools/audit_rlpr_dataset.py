"""Audit and validate the RLPR dataset structure.

This script inspects the local RLPR dataset release, validates the expected
per-sample files, summarizes image and label statistics, and exports a JSON
report for downstream engineering work.

The audit is intentionally lightweight so it can run on low-resource Windows
development machines before heavier training dependencies are installed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image


EXPECTED_FRAME_NAMES: tuple[str, ...] = tuple(f"{index:02d}.png" for index in range(1, 32))
REQUIRED_SAMPLE_FILES: tuple[str, ...] = (
    "Pseudo_GT.png",
    "Pseudo_GT_ROI.png",
    "SR_ROI.png",
    "Select_ROI/coordinate/Roi_coordinate.txt",
    "Homography_Transformation/coordinate/HP_coordinate.txt",
    "Homography_Transformation/coordinate/LP_coordinate.txt",
)


@dataclass(frozen=True)
class SampleAudit:
    """Container for per-sample audit data."""

    sample_id: str
    frame_count: int
    frame_size: tuple[int, int]
    pseudo_gt_size: tuple[int, int]
    pseudo_gt_roi_size: tuple[int, int]
    sr_roi_size: tuple[int, int]
    label: str
    label_length: int
    pseudo_gt_scale_x: float
    pseudo_gt_scale_y: float
    roi_scale_x: float
    roi_scale_y: float
    missing_files: list[str]
    frame_name_mismatches: list[str]


@dataclass(frozen=True)
class DatasetAudit:
    """Container for dataset-level audit results."""

    dataset_root: str
    sample_count: int
    label_count: int
    samples_with_errors: int
    missing_label_samples: list[str]
    frame_count_distribution: dict[str, int]
    frame_size_distribution: dict[str, int]
    pseudo_gt_size_distribution: dict[str, int]
    pseudo_gt_roi_size_distribution: dict[str, int]
    sr_roi_size_distribution: dict[str, int]
    label_length_distribution: dict[str, int]
    total_label_characters: int
    frame_width_stats: dict[str, float]
    frame_height_stats: dict[str, float]
    pseudo_gt_roi_width_stats: dict[str, float]
    pseudo_gt_roi_height_stats: dict[str, float]
    pseudo_gt_exact_4x_samples: int
    pseudo_gt_non_4x_samples: list[dict[str, Any]]
    roi_and_sr_size_mismatch_samples: list[str]
    samples: list[SampleAudit]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Audit the RLPR dataset release.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("Realistic License Plate Restoration and Recognition Dataset (RLPR)"),
        help="Path to the RLPR dataset root.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the audit as JSON.",
    )
    return parser.parse_args()


def read_image_size(image_path: Path) -> tuple[int, int]:
    """Read image size without loading full pixel arrays into memory.

    Args:
        image_path: Path to the image file.

    Returns:
        Image size as ``(width, height)``.
    """

    with Image.open(image_path) as image:
        return image.size


def summarize_numeric(values: list[int]) -> dict[str, float]:
    """Compute simple descriptive statistics for a numeric series."""

    return {
        "min": float(min(values)),
        "mean": float(mean(values)),
        "median": float(median(values)),
        "max": float(max(values)),
    }


def counter_to_string_dict(counter: Counter[Any]) -> dict[str, int]:
    """Convert a counter to a JSON-friendly string-key dictionary."""

    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def load_labels(labels_path: Path) -> list[str]:
    """Load raw labels in sample order.

    The release stores one plate string per line. Spaces inside a line are part
    of the plate formatting and should not be used as a structural delimiter.
    """

    return [line.rstrip("\n") for line in labels_path.read_text(encoding="utf-8").splitlines()]


def audit_sample(sample_dir: Path, label: str) -> SampleAudit:
    """Audit a single sample directory."""

    plate_crop_dir = sample_dir / "Plate_crop"
    frame_paths = sorted(plate_crop_dir.glob("*.png"))
    frame_names = [path.name for path in frame_paths]

    missing_files = [
        relative_path
        for relative_path in REQUIRED_SAMPLE_FILES
        if not (sample_dir / relative_path).exists()
    ]

    frame_name_mismatches = [
        expected_name
        for expected_name in EXPECTED_FRAME_NAMES
        if expected_name not in frame_names
    ]

    if not frame_paths:
        raise FileNotFoundError(f"No frame images found in {plate_crop_dir}.")

    frame_size = read_image_size(sample_dir / "Plate_crop" / "16.png")
    pseudo_gt_size = read_image_size(sample_dir / "Pseudo_GT.png")
    pseudo_gt_roi_size = read_image_size(sample_dir / "Pseudo_GT_ROI.png")
    sr_roi_size = read_image_size(sample_dir / "SR_ROI.png")

    normalized_label = label.replace(" ", "")

    return SampleAudit(
        sample_id=sample_dir.name,
        frame_count=len(frame_paths),
        frame_size=frame_size,
        pseudo_gt_size=pseudo_gt_size,
        pseudo_gt_roi_size=pseudo_gt_roi_size,
        sr_roi_size=sr_roi_size,
        label=label,
        label_length=len(normalized_label),
        pseudo_gt_scale_x=round(pseudo_gt_size[0] / frame_size[0], 4),
        pseudo_gt_scale_y=round(pseudo_gt_size[1] / frame_size[1], 4),
        roi_scale_x=round(pseudo_gt_roi_size[0] / frame_size[0], 4),
        roi_scale_y=round(pseudo_gt_roi_size[1] / frame_size[1], 4),
        missing_files=missing_files,
        frame_name_mismatches=frame_name_mismatches,
    )


def audit_dataset(dataset_root: Path) -> DatasetAudit:
    """Audit the full RLPR dataset directory."""

    labels_path = dataset_root / "Label" / "Labels.txt"
    samples_root = dataset_root / "Dataset"

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not samples_root.exists():
        raise FileNotFoundError(f"Sample root not found: {samples_root}")

    labels = load_labels(labels_path)
    sample_dirs = sorted(path for path in samples_root.iterdir() if path.is_dir())

    sample_audits: list[SampleAudit] = []
    missing_label_samples: list[str] = []

    for index, sample_dir in enumerate(sample_dirs):
        if index >= len(labels):
            missing_label_samples.append(sample_dir.name)
            label = ""
        else:
            label = labels[index]
        sample_audits.append(audit_sample(sample_dir=sample_dir, label=label))

    frame_counts = Counter(sample.frame_count for sample in sample_audits)
    frame_sizes = Counter(sample.frame_size for sample in sample_audits)
    pseudo_gt_sizes = Counter(sample.pseudo_gt_size for sample in sample_audits)
    pseudo_gt_roi_sizes = Counter(sample.pseudo_gt_roi_size for sample in sample_audits)
    sr_roi_sizes = Counter(sample.sr_roi_size for sample in sample_audits)
    label_lengths = Counter(sample.label_length for sample in sample_audits)

    frame_widths = [sample.frame_size[0] for sample in sample_audits]
    frame_heights = [sample.frame_size[1] for sample in sample_audits]
    roi_widths = [sample.pseudo_gt_roi_size[0] for sample in sample_audits]
    roi_heights = [sample.pseudo_gt_roi_size[1] for sample in sample_audits]

    pseudo_gt_non_4x_samples = [
        {
            "sample_id": sample.sample_id,
            "frame_size": list(sample.frame_size),
            "pseudo_gt_size": list(sample.pseudo_gt_size),
            "scale_x": sample.pseudo_gt_scale_x,
            "scale_y": sample.pseudo_gt_scale_y,
        }
        for sample in sample_audits
        if sample.pseudo_gt_scale_x != 4.0 or sample.pseudo_gt_scale_y != 4.0
    ]

    roi_and_sr_size_mismatch_samples = [
        sample.sample_id
        for sample in sample_audits
        if sample.pseudo_gt_roi_size != sample.sr_roi_size
    ]

    samples_with_errors = sum(
        1
        for sample in sample_audits
        if sample.missing_files or sample.frame_name_mismatches or sample.frame_count != 31
    )

    return DatasetAudit(
        dataset_root=str(dataset_root.resolve()),
        sample_count=len(sample_dirs),
        label_count=len(labels),
        samples_with_errors=samples_with_errors,
        missing_label_samples=missing_label_samples,
        frame_count_distribution=counter_to_string_dict(frame_counts),
        frame_size_distribution=counter_to_string_dict(frame_sizes),
        pseudo_gt_size_distribution=counter_to_string_dict(pseudo_gt_sizes),
        pseudo_gt_roi_size_distribution=counter_to_string_dict(pseudo_gt_roi_sizes),
        sr_roi_size_distribution=counter_to_string_dict(sr_roi_sizes),
        label_length_distribution=counter_to_string_dict(label_lengths),
        total_label_characters=sum(sample.label_length for sample in sample_audits),
        frame_width_stats=summarize_numeric(frame_widths),
        frame_height_stats=summarize_numeric(frame_heights),
        pseudo_gt_roi_width_stats=summarize_numeric(roi_widths),
        pseudo_gt_roi_height_stats=summarize_numeric(roi_heights),
        pseudo_gt_exact_4x_samples=len(sample_audits) - len(pseudo_gt_non_4x_samples),
        pseudo_gt_non_4x_samples=pseudo_gt_non_4x_samples,
        roi_and_sr_size_mismatch_samples=roi_and_sr_size_mismatch_samples,
        samples=sample_audits,
    )


def main() -> None:
    """Run the audit and optionally persist it as JSON."""

    args = parse_args()
    audit = audit_dataset(args.dataset_root)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(asdict(audit), indent=2),
            encoding="utf-8",
        )

    summary = {
        "dataset_root": audit.dataset_root,
        "sample_count": audit.sample_count,
        "label_count": audit.label_count,
        "samples_with_errors": audit.samples_with_errors,
        "frame_count_distribution": audit.frame_count_distribution,
        "label_length_distribution": audit.label_length_distribution,
        "total_label_characters": audit.total_label_characters,
        "frame_width_stats": audit.frame_width_stats,
        "frame_height_stats": audit.frame_height_stats,
        "pseudo_gt_roi_width_stats": audit.pseudo_gt_roi_width_stats,
        "pseudo_gt_roi_height_stats": audit.pseudo_gt_roi_height_stats,
        "pseudo_gt_exact_4x_samples": audit.pseudo_gt_exact_4x_samples,
        "pseudo_gt_non_4x_count": len(audit.pseudo_gt_non_4x_samples),
        "roi_and_sr_size_mismatch_count": len(audit.roi_and_sr_size_mismatch_samples),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
