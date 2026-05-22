"""
OCR-related metrics module for license plate restoration evaluation.

Provides:
- Character Error Rate (CER)
- Sequence Accuracy (exact match on plate text)
- Hooks for PARSeq OCR integration (Phase 6+)
"""

from typing import Dict, List, Tuple, Optional, Callable
try:
    import editdistance
except ImportError:
    editdistance = None


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Pure-python implementation of Levenshtein distance as fallback."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _get_edit_distance(s1: str, s2: str) -> int:
    """Helper that uses editdistance if available, otherwise pure-python fallback."""
    if editdistance is not None:
        try:
            return editdistance.eval(s1, s2)
        except Exception:
            pass
    return _levenshtein_distance(s1, s2)


class CharacterErrorRate:
    """Computes Character Error Rate (CER) metric."""

    @staticmethod
    def compute_cer(predictions: List[str], ground_truth: List[str]) -> float:
        """
        Compute Character Error Rate from OCR predictions.

        CER = (S + D + I) / N

        Where:
            S = substitutions
            D = deletions
            I = insertions
            N = total characters in ground truth

        Args:
            predictions: List of predicted plate texts
            ground_truth: List of ground truth plate texts

        Returns:
            CER value in [0, 1] (0 = perfect, 1 = completely wrong)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        total_errors = 0
        total_chars = 0

        for pred, gt in zip(predictions, ground_truth):
            # Use edit distance to compute S + D + I
            errors = _get_edit_distance(pred, gt)
            total_errors += errors
            total_chars += len(gt)

        if total_chars == 0:
            return 0.0

        cer = total_errors / total_chars
        return cer

    @staticmethod
    def compute_cer_per_sample(predictions: List[str], ground_truth: List[str]) -> List[float]:
        """
        Compute CER for each sample individually.

        Args:
            predictions: List of predicted plate texts
            ground_truth: List of ground truth plate texts

        Returns:
            List of CER values [0, 1] per sample
        """
        cers = []
        for pred, gt in zip(predictions, ground_truth):
            errors = _get_edit_distance(pred, gt)
            gt_len = len(gt)
            cer = errors / gt_len if gt_len > 0 else 0.0
            cers.append(cer)
        return cers

    @staticmethod
    def format_cer_breakdown(predictions: List[str], ground_truth: List[str]) -> Dict[str, any]:
        """
        Compute detailed CER breakdown with substitutions, deletions, insertions.

        Args:
            predictions: List of predicted plate texts
            ground_truth: List of ground truth plate texts

        Returns:
            Dictionary with:
                - cer: Overall CER
                - total_errors: Total edit distance
                - total_chars: Total characters in GT
                - avg_errors_per_sample: Average edit distance per prediction
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        total_errors = 0
        total_chars = 0
        error_counts = []

        for pred, gt in zip(predictions, ground_truth):
            errors = _get_edit_distance(pred, gt)
            total_errors += errors
            total_chars += len(gt)
            error_counts.append(errors)

        cer = total_errors / total_chars if total_chars > 0 else 0.0
        avg_errors = sum(error_counts) / len(error_counts) if error_counts else 0.0

        return {
            "cer": cer,
            "total_errors": total_errors,
            "total_chars": total_chars,
            "avg_errors_per_sample": avg_errors,
            "num_samples": len(predictions),
        }


class SequenceAccuracy:
    """Computes exact match accuracy for plate sequences."""

    @staticmethod
    def compute_sequence_accuracy(predictions: List[str], ground_truth: List[str]) -> float:
        """
        Compute fraction of perfectly recognized plates.

        Args:
            predictions: List of predicted plate texts
            ground_truth: List of ground truth plate texts

        Returns:
            Accuracy in [0, 1] (1 = all plates perfect)
        """
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have same length")

        correct = sum(1 for pred, gt in zip(predictions, ground_truth) if pred == gt)
        accuracy = correct / len(ground_truth)
        return accuracy

    @staticmethod
    def compute_per_position_accuracy(predictions: List[str], ground_truth: List[str]) -> List[float]:
        """
        Compute accuracy for each character position.

        Useful for identifying which positions are harder to recognize.

        Args:
            predictions: List of predicted plate texts
            ground_truth: List of ground truth plate texts

        Returns:
            List of per-position accuracies
        """
        if not ground_truth:
            return []

        # Find max length
        max_len = max(len(gt) for gt in ground_truth)
        per_position_correct = [0] * max_len
        per_position_total = [0] * max_len

        for pred, gt in zip(predictions, ground_truth):
            for i, (p_char, gt_char) in enumerate(zip(pred, gt)):
                per_position_total[i] += 1
                if p_char == gt_char:
                    per_position_correct[i] += 1

            # Handle extra characters in prediction
            for i in range(len(gt), len(pred)):
                per_position_total[i] += 1

            # Handle missing characters (already counted as incorrect above)

        per_position_accuracy = [
            correct / total if total > 0 else 0.0
            for correct, total in zip(per_position_correct, per_position_total)
        ]
        return per_position_accuracy


class OCRMetricHooks:
    """
    Infrastructure for integrating external OCR engines.

    Phase 6+ will wire PARSeq OCR here.
    """

    def __init__(self, ocr_engine: Optional[Callable] = None):
        """
        Initialize OCR hooks.

        Args:
            ocr_engine: Optional callable that takes image tensor and returns plate text.
                       Signature: (image: torch.Tensor) -> str
                       Example: PARSeq model from Phase 6
        """
        self.ocr_engine = ocr_engine
        self.is_phase6_ready = False

    def register_ocr_engine(self, ocr_engine: Callable):
        """
        Register external OCR engine (e.g., PARSeq from Phase 6).

        Args:
            ocr_engine: Callable(image: torch.Tensor) -> str
        """
        self.ocr_engine = ocr_engine
        self.is_phase6_ready = True
        print("✓ PARSeq OCR engine registered. CER computation now uses live OCR.")

    def predict_plates(self, images: list) -> List[str]:
        """
        Predict plate texts using registered OCR engine.

        Args:
            images: List of restored image tensors (C, H, W) or (B, C, H, W)

        Returns:
            List of predicted plate texts

        Raises:
            RuntimeError if no OCR engine registered
        """
        if self.ocr_engine is None:
            raise RuntimeError(
                "No OCR engine registered. In Phase 6, wire PARSeq OCR via "
                "register_ocr_engine(). For now, use ground truth labels."
            )

        predictions = []
        for image in images:
            pred_text = self.ocr_engine(image)
            predictions.append(pred_text)
        return predictions

    def compute_live_cer(self, images: list, ground_truth: List[str]) -> Dict[str, float]:
        """
        Compute CER using live OCR on restored images.

        This is the Phase 6+ workflow. For now, raises error (use compute_mock_cer instead).

        Args:
            images: List of restored image tensors
            ground_truth: List of ground truth plate texts

        Returns:
            Dictionary with CER metrics

        Raises:
            RuntimeError if PARSeq not registered
        """
        if not self.is_phase6_ready:
            raise RuntimeError(
                "PARSeq OCR not registered. Phase 6 will wire this. "
                "For Phase 5.1 evaluation, use compute_mock_cer()."
            )

        predictions = self.predict_plates(images)
        cer = CharacterErrorRate.compute_cer(predictions, ground_truth)
        seq_acc = SequenceAccuracy.compute_sequence_accuracy(predictions, ground_truth)

        return {
            "cer_live": cer,
            "sequence_accuracy_live": seq_acc,
            "predictions": predictions,
        }

    @staticmethod
    def compute_mock_cer(restored_quality_metrics: Dict[str, float], ground_truth_count: int) -> Dict[str, float]:
        """
        Compute mock CER based on restoration quality.

        This is a placeholder for Phase 5.1. In Phase 6, this will be replaced by live OCR.

        The mock CER is estimated as: CER_mock = 1.0 - normalize(SSIM)
        This is a rough proxy: better restoration → lower CER.

        Args:
            restored_quality_metrics: Dictionary with 'ssim', 'psnr', etc.
            ground_truth_count: Number of ground truth samples (for normalization)

        Returns:
            Dictionary with mock CER metrics (for logging purposes)
        """
        ssim = restored_quality_metrics.get("ssim", 0.0)
        psnr = restored_quality_metrics.get("psnr", 0.0)

        # Rough estimation: SSIM in [0,1], CER in [0,1]
        # Better SSIM → lower CER
        mock_cer = 1.0 - ssim  # Very rough proxy

        # Clip to [0, 1]
        mock_cer = max(0.0, min(1.0, mock_cer))

        # Mock sequence accuracy: optimistic if PSNR high
        mock_seq_acc = min(1.0, max(0.0, psnr / 30.0))  # Rough: PSNR > 30 → high accuracy

        return {
            "cer_mock": mock_cer,
            "sequence_accuracy_mock": mock_seq_acc,
            "note": "Mock CER from restoration metrics (SSIM/PSNR proxy). "
            "Real CER computed in Phase 6 with PARSeq OCR.",
        }


class OCRMetricsEvaluator:
    """Unified evaluator for OCR-related metrics."""

    def __init__(self, use_mock_cer: bool = True):
        """
        Initialize evaluator.

        Args:
            use_mock_cer: If True, use mock CER (Phase 5.1). If False, require live OCR (Phase 6+).
        """
        self.use_mock_cer = use_mock_cer
        self.ocr_hooks = OCRMetricHooks()

    def evaluate(
        self,
        predictions: List[str],
        ground_truth: List[str],
        restoration_metrics: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Evaluate OCR metrics.

        Args:
            predictions: Predicted plate texts (or mock predictions if using mock CER)
            ground_truth: Ground truth plate texts
            restoration_metrics: Optional dict with SSIM, PSNR for mock CER

        Returns:
            Dictionary with CER, sequence accuracy, and breakdown
        """
        results = {}

        # Compute CER and sequence accuracy
        cer = CharacterErrorRate.compute_cer(predictions, ground_truth)
        seq_acc = SequenceAccuracy.compute_sequence_accuracy(predictions, ground_truth)
        cer_breakdown = CharacterErrorRate.format_cer_breakdown(predictions, ground_truth)

        results["cer"] = cer
        results["sequence_accuracy"] = seq_acc
        results.update(cer_breakdown)

        # Add mock CER if restoration metrics provided
        if self.use_mock_cer and restoration_metrics is not None:
            mock_metrics = OCRMetricHooks.compute_mock_cer(restoration_metrics, len(ground_truth))
            results.update(mock_metrics)

        return results


if __name__ == "__main__":
    # Example usage
    gt = ["AB1234", "CD5678", "EF9012"]
    pred = ["AB1234", "CD5679", "EF9012"]  # One typo

    cer = CharacterErrorRate.compute_cer(pred, gt)
    seq_acc = SequenceAccuracy.compute_sequence_accuracy(pred, gt)

    print(f"CER: {cer:.3f}")  # 0.167 (1 error out of 6 chars per plate * 3)
    print(f"Sequence Accuracy: {seq_acc:.3f}")  # 0.667 (2 out of 3 correct)

    breakdown = CharacterErrorRate.format_cer_breakdown(pred, gt)
    print(f"CER Breakdown: {breakdown}")

    per_position_acc = SequenceAccuracy.compute_per_position_accuracy(pred, gt)
    print(f"Per-position accuracy: {per_position_acc}")

    # Mock CER example
    mock_result = OCRMetricHooks.compute_mock_cer({"ssim": 0.8, "psnr": 25.0}, len(gt))
    print(f"Mock CER: {mock_result}")
