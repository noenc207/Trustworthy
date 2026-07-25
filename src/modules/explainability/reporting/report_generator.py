import json
import os
from pathlib import Path

from src.modules.explainability.config import ExplainabilityConfig
from src.modules.explainability.interfaces import ReportGeneratorInterface
from src.modules.explainability.reporting.clinical_interpreter import ClinicalInterpreter
from src.modules.explainability.result import ClinicalReport, ExplainabilityResult


class ClinicalReportGenerator(ReportGeneratorInterface):
    """Generates JSON, Markdown, and PDF (stub) clinical reports."""

    def generate(self, result: ExplainabilityResult, config: ExplainabilityConfig) -> ClinicalReport:
        is_reliable = bool(result.overall_quality_score > 70)
        interpreter = ClinicalInterpreter()
        interpretation = interpreter.interpret(result)

        metrics_breakdown = {}
        if result.medical_metrics:
            metrics_breakdown["medical"] = result.medical_metrics.metrics_score
        if result.faithfulness:
            metrics_breakdown["faithfulness"] = result.faithfulness.faithfulness_score
        if result.stability:
            metrics_breakdown["stability"] = result.stability.stability_score
        if result.sanity:
            metrics_breakdown["sanity"] = result.sanity.sanity_score

        return ClinicalReport(
            executive_summary=f"Explainability execution completed in {result.inference_time:.2f}s.",
            clinical_interpretation=interpretation,
            overall_quality_score=result.overall_quality_score,
            is_reliable=is_reliable,
            metrics_breakdown=metrics_breakdown,
            warnings=result.warnings,
            recommendations=["Review bounding boxes if background leakage is high."]
        )

    def export(self, report: ClinicalReport, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = Path(output_dir)

        # JSON
        import dataclasses
        with open(out_path / "clinical_report.json", "w", encoding="utf-8") as f:
            json.dump(dataclasses.asdict(report), f, indent=4)

        # Markdown
        with open(out_path / "summary.md", "w", encoding="utf-8") as f:
            f.write("# Explainability Report\n")
            f.write(f"Quality Score: {report.overall_quality_score:.2f}\n")
            f.write(f"Reliable: {report.is_reliable}\n\n")
            f.write("## Warnings\n")
            for w in report.warnings:
                f.write(f"- {w}\n")

        # Simple HTML stub
        with open(out_path / "clinical_report.html", "w", encoding="utf-8") as f:
            f.write(f"<html><body><h1>Clinical Report</h1><p>Score: {report.overall_quality_score:.2f}</p></body></html>")
