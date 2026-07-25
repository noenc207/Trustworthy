from typing import Any

from src.modules.explainability.result import CECIResult


class ConsistencyReportGenerator:
    def generate(self, ceci_result: CECIResult, conflicts: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate a summarized report of the consistency evaluation."""
        return {
            "ceci_score": ceci_result.ceci_score,
            "confidence_interval": ceci_result.ceci_stats.ci_95,
            "recommendation": ceci_result.recommendation,
            "conflicts": conflicts,
            "provenance": {
                "origin": ceci_result.provenance.origin,
                "version": ceci_result.provenance.version,
                "timestamp": ceci_result.provenance.timestamp
            }
        }
