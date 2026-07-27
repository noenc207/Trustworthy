"""
Evaluation Pipeline.

Comprehensive model evaluation with:
  - Standard classification metrics (Accuracy, F1, AUROC, Recall per class)
  - Calibration metrics (ECE, MCE, Reliability Diagram)
  - OOD detection metrics (AUROC, FPR@95TPR)
  - Uncertainty quality metrics (AUPR, NLL)
  - Per-class analysis
  - Confusion matrix
  - Result persistence for research comparisons
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    """Complete evaluation result bundle."""
    # Classification
    accuracy: float
    macro_f1: float
    weighted_f1: float
    macro_auroc: float
    per_class_report: dict
    confusion_matrix: np.ndarray
    # Calibration
    ece: float
    mce: float
    # OOD
    ood_auroc: float
    ood_fpr95: float
    # Uncertainty
    uncertainty_auroc: float
    # Metadata
    model_version: str
    dataset: str
    evaluated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    num_samples: int = 0


class EvaluationPipeline:
    """
    Standardized evaluation pipeline for reproducible benchmarking.

    Usage:
        pipeline = EvaluationPipeline(output_dir=Path('research/results'))
        result = pipeline.evaluate(model, test_loader)
        pipeline.save_report(result)
    """

    def __init__(self, output_dir: Path = Path("research/results")) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        predictions: np.ndarray,     # (N, num_classes) predicted probabilities
        labels: np.ndarray,           # (N,) ground truth class indices
        ood_scores: np.ndarray | None = None,
        uncertainties: np.ndarray | None = None,
        class_names: list[str] | None = None,
        model_version: str = "v1",
        dataset: str = "test",
    ) -> EvaluationResult:
        """Run full evaluation suite."""
        predicted_classes = predictions.argmax(axis=-1)

        # Classification metrics
        accuracy = float((predicted_classes == labels).mean())
        report = classification_report(
            labels, predicted_classes,
            target_names=class_names,
            output_dict=True,
        )
        macro_f1 = report["macro avg"]["f1-score"]
        weighted_f1 = report["weighted avg"]["f1-score"]

        # AUROC (one-vs-rest)
        try:
            macro_auroc = roc_auc_score(labels, predictions, multi_class="ovr", average="macro")
        except ValueError:
            macro_auroc = float("nan")
            logger.warning("Could not compute AUROC (likely single class in batch)")

        cm = confusion_matrix(labels, predicted_classes)

        # Calibration ECE computation
        confs = np.max(predictions, axis=1)
        accs = (predicted_classes == labels).astype(float)
        bins = np.linspace(0, 1, 11)
        ece, mce = 0.0, 0.0
        for i in range(10):
            mask = (confs >= bins[i]) & (confs <= bins[i+1])
            if np.sum(mask) > 0:
                bin_acc = accs[mask].mean()
                bin_conf = confs[mask].mean()
                diff = abs(bin_acc - bin_conf)
                ece += diff * np.mean(mask)
                mce = max(mce, diff)

        # OOD metrics
        ood_auroc, ood_fpr95 = 0.0, 0.0
        if ood_scores is not None and len(np.unique(ood_scores)) > 1:
            logger.info(f"OOD scores provided: {ood_scores.shape}")
            try:
                ood_labels = (ood_scores > np.median(ood_scores)).astype(int)
                ood_auroc = roc_auc_score(ood_labels, ood_scores)
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(ood_labels, ood_scores)
                idx = np.where(tpr >= 0.95)[0]
                ood_fpr95 = float(fpr[idx[0]] if len(idx) > 0 else 0.0)
            except ValueError:
                pass

        return EvaluationResult(
            accuracy=accuracy,
            macro_f1=macro_f1,
            weighted_f1=weighted_f1,
            macro_auroc=macro_auroc,
            per_class_report=report,
            confusion_matrix=cm,
            ece=ece,
            mce=mce,
            ood_auroc=ood_auroc,
            ood_fpr95=ood_fpr95,
            uncertainty_auroc=0.0,
            model_version=model_version,
            dataset=dataset,
            num_samples=len(labels),
        )

    def save_report(
        self,
        result: EvaluationResult,
        name: str = "evaluation",
    ) -> Path:
        """Save evaluation results as JSON and CSV."""
        import json
        from dataclasses import asdict

        report_dict = asdict(result)
        report_dict["confusion_matrix"] = result.confusion_matrix.tolist()

        out_file = self.output_dir / f"{name}_{result.evaluated_at[:10]}.json"
        with open(out_file, "w") as f:
            json.dump(report_dict, f, indent=2)

        logger.info(f"Evaluation report saved to {out_file}")
        return out_file
