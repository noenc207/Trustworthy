"""Claim consistency validator."""

from typing import Any


class ClaimConsistencyValidator:
    """Validates claims against available evidence."""

    @staticmethod
    def validate_claim(claim: str, evidence: dict[str, Any]) -> str:
        """Validates a specific claim.

        Args:
            claim (str): The claim to validate.
            evidence (Dict[str, Any]): The available evidence.

        Returns:
            str: "Supported" or "Unsupported Claim: <reason>".
        """
        claim_lower = claim.lower()

        if "significant" in claim_lower:
            p_value = evidence.get("p_value")
            if p_value is None:
                return "Unsupported Claim: Missing p-value for significance claim."
            if not isinstance(p_value, (int, float)):
                return "Unsupported Claim: Invalid p-value format."
            if p_value >= 0.05:
                return f"Unsupported Claim: p-value {p_value} >= 0.05 is not significant."

        if "sota" in claim_lower or "state of the art" in claim_lower or "state-of-the-art" in claim_lower:
            benchmark = evidence.get("benchmark")
            if not benchmark:
                return "Unsupported Claim: Missing benchmark for SOTA claim."

        return "Supported"
