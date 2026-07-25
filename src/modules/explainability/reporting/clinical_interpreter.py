from src.modules.explainability.result import ExplainabilityResult


class ClinicalInterpreter:
    """Generates human-readable clinical summaries based on metrics."""

    @staticmethod
    def interpret(result: ExplainabilityResult) -> str:
        text = "Clinical Interpretation:\n"

        # Localization
        if result.medical_metrics:
            mm = result.medical_metrics
            if mm.lesion_coverage > 80:
                text += "- The model exhibits excellent focus on the lesion area.\n"
            elif mm.lesion_coverage < 20:
                text += "- The model primarily focuses on background or irrelevant artifacts.\n"

            if mm.compactness > 80:
                text += "- The attention map is highly concentrated.\n"
            else:
                text += "- The attention is dispersed across multiple regions.\n"

        # Faithfulness
        if result.faithfulness:
            f = result.faithfulness
            if f.faithfulness_score > 80:
                text += "- Faithfulness metrics indicate the highlighted regions strongly drive the prediction.\n"
            else:
                text += "- WARNING: The highlighted regions do not fully explain the model's confidence.\n"

        # Stability
        if result.stability:
            if result.stability.stability_score > 80:
                text += "- The explanation is highly robust to image perturbations.\n"
            else:
                text += "- WARNING: The explanation is sensitive to noise or rotation.\n"

        # Sanity
        if result.sanity:
            if not result.sanity.weight_randomization_passed:
                text += "- CRITICAL WARNING: Failed Adebayo Sanity Check. Explanation may not depend on learned weights.\n"

        if result.overall_quality_score > 80:
            text += "\nOverall Explanation Reliability: HIGH"
        elif result.overall_quality_score > 50:
            text += "\nOverall Explanation Reliability: MODERATE"
        else:
            text += "\nOverall Explanation Reliability: LOW"

        return text
