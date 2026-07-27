import time
from typing import Any

import numpy as np

from src.modules.calibration.config import CalibrationConfig
from src.modules.calibration.exceptions import UnsupportedCalibrationAlgorithmError
from src.modules.calibration.interfaces import CalibrationEngine, CalibrationStrategy
from src.modules.calibration.metrics import (
    compute_adaptive_ece,
    compute_brier_score,
    compute_ece_mce,
    compute_nll,
    compute_summary_metrics,
)
from src.modules.calibration.result import CalibrationResult
from src.modules.calibration.strategies.histogram import HistogramBinningStrategy
from src.modules.calibration.strategies.isotonic import IsotonicRegressionStrategy
from src.modules.calibration.strategies.platt import PlattScalingStrategy
from src.modules.calibration.strategies.temperature import TemperatureScalingStrategy
from src.modules.calibration.visualization import (
    plot_confidence_histogram,
    plot_reliability_diagram,
)
from src.modules.classifier.result import PredictionResult


class DefaultCalibrationEngine(CalibrationEngine):
    """Production-Grade Orchestrator for Confidence Calibration."""

    def __init__(self, config: CalibrationConfig):
        self.config = config
        self.strategy = self._load_strategy()

    def _load_strategy(self) -> CalibrationStrategy:
        algo = self.config.algorithm.lower()
        if algo == "temperature":
            return TemperatureScalingStrategy(config=self.config)
        elif algo == "platt":
            return PlattScalingStrategy(a=getattr(self.config, 'platt_a', 1.0), b=getattr(self.config, 'platt_b', 0.0))
        elif algo == "isotonic":
            return IsotonicRegressionStrategy()
        elif algo == "histogram":
            return HistogramBinningStrategy()
        else:
            raise UnsupportedCalibrationAlgorithmError(f"Unsupported algorithm: {self.config.algorithm}")

    def _arrayify_probs(self, classification_result: PredictionResult) -> np.ndarray:
        return np.array([list(classification_result.probabilities.values())])

    def evaluate(
        self,
        classification_result: PredictionResult,
        raw_logits: Any = None,
        true_labels: Any = None
    ) -> CalibrationResult:
        start_time = time.time()
        orig_conf = classification_result.confidence

        try:
            orig_probs = self._arrayify_probs(classification_result)
            labels_arr = np.array([true_labels]) if true_labels is not None else np.array([classification_result.predicted_index])

            # 1. Baseline Metrics
            n_bins = self.config.num_bins
            ece_before, mce_before, _ = compute_ece_mce(orig_probs, labels_arr, n_bins)
            aece_before = compute_adaptive_ece(orig_probs, labels_arr, n_bins)
            brier_before = compute_brier_score(orig_probs, labels_arr)
            nll_before = compute_nll(orig_probs, labels_arr)
            summ_before = compute_summary_metrics(orig_probs, labels_arr)

            # 2. Apply Calibration
            metrics = self.strategy.calibrate(classification_result, raw_logits)
            calibrated_conf = metrics.get("calibrated_confidence", orig_conf)
            calibrated_probs_dict = metrics.get("calibrated_probabilities", classification_result.probabilities)

            calib_probs = np.array([list(calibrated_probs_dict.values())])

            # 3. Post-Calibration Metrics
            ece_after, mce_after, _ = compute_ece_mce(calib_probs, labels_arr, n_bins)
            aece_after = compute_adaptive_ece(calib_probs, labels_arr, n_bins)
            brier_after = compute_brier_score(calib_probs, labels_arr)
            nll_after = compute_nll(calib_probs, labels_arr)
            summ_after = compute_summary_metrics(calib_probs, labels_arr)

            # 4. Strict Rollback Evaluation
            is_calibrated = False
            rollback = False
            warnings = []
            improved = False

            eps = 1e-6
            if (ece_after <= ece_before + eps) and (brier_after <= brier_before + eps) and (nll_after <= nll_before + eps):
                is_calibrated = True
                improved = (ece_after < ece_before) or (brier_after < brier_before) or (nll_after < nll_before)
            else:
                rollback = True
                warnings.append(f"Calibration worsened metrics (ECE: {ece_before:.4f}->{ece_after:.4f}, NLL: {nll_before:.4f}->{nll_after:.4f}). Rolling back.")
                calibrated_conf = orig_conf
                calib_probs = orig_probs
                ece_after, mce_after, aece_after = ece_before, mce_before, aece_before
                brier_after, nll_after = brier_before, nll_before
                summ_after = summ_before

            history = metrics.get("history", {})
            iterations = history.get("iterations", 0)
            opt_time = history.get("execution_time", 0.0)
            temp = history.get("temperature", 1.0) if not rollback else 1.0
            optimizer = getattr(self.config, 'optimizer', 'none')

            # 5. Visualizations (if enabled and batch > 1)
            # Typically single instances won't generate meaningful histograms, but requested by spec.
            if self.config.generate_diagrams and len(labels_arr) > 1:
                out_dir = str(self.config.output_dir)
                plot_reliability_diagram(orig_probs, labels_arr, f"{out_dir}/reliability_before.png", n_bins, f"Reliability Before (ECE: {ece_before:.4f})")
                plot_reliability_diagram(calib_probs, labels_arr, f"{out_dir}/reliability_after.png", n_bins, f"Reliability After (ECE: {ece_after:.4f})")
                plot_confidence_histogram(orig_probs, f"{out_dir}/confidence_histogram_before.png", n_bins)
                plot_confidence_histogram(calib_probs, f"{out_dir}/confidence_histogram_after.png", n_bins)

            exec_time = time.time() - start_time

            res = CalibrationResult(
                temperature=temp,
                optimizer=optimizer,
                iterations=iterations,
                training_time=opt_time,

                ece_before=ece_before,
                ece_after=ece_after,
                adaptive_ece_before=aece_before,
                adaptive_ece_after=aece_after,
                mce_before=mce_before,
                mce_after=mce_after,
                brier_before=brier_before,
                brier_after=brier_after,
                nll_before=nll_before,
                nll_after=nll_after,

                confidence_before=orig_conf,
                confidence_after=calibrated_conf,

                improved=improved,
                rollback=rollback,
                is_calibrated=is_calibrated,
                execution_time=exec_time,

                metrics={
                    "mean_confidence_before": summ_before["mean_confidence"],
                    "mean_confidence_after": summ_after["mean_confidence"],
                    "confidence_gap_before": summ_before["confidence_gap"],
                    "confidence_gap_after": summ_after["confidence_gap"]
                },
                warnings=warnings,
                metadata={
                    "dataset_statistics": {
                        "samples": len(labels_arr),
                        "classes": orig_probs.shape[1]
                    }
                }
            )

            # Generate JSON
            if self.config.generate_diagrams and len(labels_arr) > 1:
                res.to_json(f"{self.config.output_dir}/report.json")

            return res

        except Exception as e:
            if self.config.fail_safe_conservative:
                return CalibrationResult(
                    temperature=1.0, optimizer="none", iterations=0, training_time=0.0,
                    ece_before=0.0, ece_after=0.0, adaptive_ece_before=0.0, adaptive_ece_after=0.0,
                    mce_before=0.0, mce_after=0.0, brier_before=0.0, brier_after=0.0,
                    nll_before=0.0, nll_after=0.0, confidence_before=orig_conf, confidence_after=orig_conf,
                    improved=False, rollback=True, is_calibrated=False, execution_time=time.time() - start_time,
                    warnings=[f"Calibration failed: {e}"]
                )
            raise e
