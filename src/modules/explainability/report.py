import json
import os
from pathlib import Path

from src.modules.explainability.result import ExplainabilityResult


class ReportGenerator:
    """Generates the JSON reporting structure for Explainability."""

    @staticmethod
    def generate(result: ExplainabilityResult, output_dir: Path) -> None:
        os.makedirs(output_dir, exist_ok=True)

        report = {
            "metadata": {
                "algorithm": result.algorithm,
                "layer_name": result.layer_name,
                "inference_time_sec": result.inference_time,
                "warnings": result.warnings
            },
            "medical_metrics": {
                "lesion_coverage": result.lesion_coverage,
                "activation_area_pct": result.activation_area_pct,
                "largest_connected_component_pct": result.largest_connected_component_pct,
                "center_of_mass": result.center_of_mass,
                "activation_compactness": result.activation_compactness,
                "peak_activation": result.peak_activation,
                "heatmap_entropy": result.heatmap_entropy,
                "attention_dispersion": result.attention_dispersion,
                "sparsity": result.sparsity,
                "focus_score": result.focus_score,
                "localization_score": result.localization_score,
                "noise_score": result.noise_score,
                "overall_quality_score": result.overall_quality_score
            },
            "activation_statistics": result.activation_statistics,
            "bounding_box": result.confidence_region
        }

        with open(output_dir / "report.json", "w") as f:
            json.dump(report, f, indent=4)
