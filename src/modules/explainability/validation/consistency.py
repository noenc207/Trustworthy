import json
import os
from typing import Any


class MetricConsistencyValidator:
    """
    Enforces cross-metric logic rules to detect contradictions among
    explainability and trustworthiness metrics.
    """
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir

    def validate(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """
        Validates logical consistency among provided metrics.
        """
        contradictions = []

        # Parse metrics with defaults
        faithfulness = float(metrics.get("Faithfulness", 0.0))
        localization = float(metrics.get("Localization", 0.0))
        ood = float(metrics.get("OOD", 0.0))
        trust = float(metrics.get("Trust", 0.0))
        sanity_failed = bool(metrics.get("Sanity Failed", False))
        ceci = float(metrics.get("CECI", 0.0))
        ceas = float(metrics.get("CEAS", 0.0))
        calibration = float(metrics.get("Calibration", 0.0))
        confidence = float(metrics.get("Confidence", 0.0))
        cf_distance = float(metrics.get("Counterfactual Distance", 0.0))
        prediction_stable = bool(metrics.get("Prediction Stable", False))
        drift = float(metrics.get("Drift", 0.0))
        failure_prob = float(metrics.get("Failure Probability", 0.0))
        heatmap_quality = float(metrics.get("Heatmap Quality", 0.0))

        # 1. Faithfulness >90 & Localization <20 -> SPURIOUS FEATURE LEARNING
        if faithfulness > 90 and localization < 20:
            contradictions.append({
                "Severity": "High",
                "Description": "SPURIOUS FEATURE LEARNING",
                "Affected Metrics": ["Faithfulness", "Localization"],
                "Recommendation": "Check if model is relying on background features instead of ROI.",
                "Confidence": 0.95
            })

        # 2. OOD >80 & Trust >70 -> INVALID TRUST
        if ood > 80 and trust > 70:
            contradictions.append({
                "Severity": "Critical",
                "Description": "INVALID TRUST",
                "Affected Metrics": ["OOD", "Trust"],
                "Recommendation": "Model is highly confident on out-of-distribution data. Review OOD detector or trust metric.",
                "Confidence": 0.90
            })

        # 3. Sanity Failed & CECI >0 -> IMPOSSIBLE STATE
        if sanity_failed and ceci > 0:
            contradictions.append({
                "Severity": "Critical",
                "Description": "IMPOSSIBLE STATE",
                "Affected Metrics": ["Sanity Failed", "CECI"],
                "Recommendation": "Explanations failed basic sanity checks yet show positive causal impact. Debug explanation generation.",
                "Confidence": 0.99
            })

        # 4. Localization >90 & CEAS <20 -> CLINICAL CONFLICT
        if localization > 90 and ceas < 20:
            contradictions.append({
                "Severity": "Medium",
                "Description": "CLINICAL CONFLICT",
                "Affected Metrics": ["Localization", "CEAS"],
                "Recommendation": "High overlap with annotations but low clinical alignment score. Check annotation quality.",
                "Confidence": 0.85
            })

        # 5. High Calibration & Low Confidence -> CONFIDENCE INCONSISTENCY
        if calibration > 80 and confidence < 20:
            contradictions.append({
                "Severity": "Medium",
                "Description": "CONFIDENCE INCONSISTENCY",
                "Affected Metrics": ["Calibration", "Confidence"],
                "Recommendation": "Model is well-calibrated but generally underconfident. Inspect calibration curve.",
                "Confidence": 0.80
            })

        # 6. Large Counterfactual Distance & Prediction Stable -> COUNTERFACTUAL CONFLICT
        if cf_distance > 80 and prediction_stable:
            contradictions.append({
                "Severity": "High",
                "Description": "COUNTERFACTUAL CONFLICT",
                "Affected Metrics": ["Counterfactual Distance", "Prediction Stable"],
                "Recommendation": "Large perturbations did not change predictions. Evaluate counterfactual generator.",
                "Confidence": 0.90
            })

        # 7. High Drift & High Trust -> TEMPORAL INCONSISTENCY
        if drift > 80 and trust > 80:
            contradictions.append({
                "Severity": "High",
                "Description": "TEMPORAL INCONSISTENCY",
                "Affected Metrics": ["Drift", "Trust"],
                "Recommendation": "Significant data drift detected but model trust remains high. Review temporal stability.",
                "Confidence": 0.85
            })

        # 8. Failure Probability High & Heatmap Quality High -> ARTIFACT CONFLICT
        if failure_prob > 80 and heatmap_quality > 80:
            contradictions.append({
                "Severity": "Medium",
                "Description": "ARTIFACT CONFLICT",
                "Affected Metrics": ["Failure Probability", "Heatmap Quality"],
                "Recommendation": "Model is likely to fail despite producing high-quality heatmaps. Investigate artifacts.",
                "Confidence": 0.85
            })

        return {
            "is_consistent": len(contradictions) == 0,
            "contradictions": contradictions
        }

    def generate_report(self, validation_results: dict[str, Any], filename: str = "consistency_validation.json") -> str:
        """
        Generates a JSON report of the consistency validation results.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(validation_results, f, indent=4)
        return filepath
