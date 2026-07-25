
from src.modules.explainability.result import ClinicalReport


class ClinicalReportGenerator:
    def __init__(self):
        self.version = "1.0"
        self.format = "json"

    def generate_report(self, alignment_score: float, morphology: dict[str, float], rules: list[str]) -> ClinicalReport:
        is_reliable = alignment_score > 0.7

        return ClinicalReport(
            executive_summary="Clinical validation completed.",
            clinical_interpretation="Findings are consistent with typical guidelines." if is_reliable else "Findings deviate from standard guidelines.",
            overall_quality_score=alignment_score * 100,
            is_reliable=is_reliable,
            metrics_breakdown=morphology,
            warnings=[] if is_reliable else ["Low alignment with clinical rules."],
            recommendations=["Review by dermatologist recommended."]
        )
