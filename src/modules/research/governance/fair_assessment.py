"""Fair assessment tools."""



class FairAssessment:
    """Provides tools for fair assessment of models."""

    @staticmethod
    def check_demographic_parity(group_outcomes: dict[str, float], threshold: float = 0.1) -> bool:
        """Checks demographic parity across groups.

        Args:
            group_outcomes (Dict[str, float]): Outcomes by group.
            threshold (float, optional): Maximum allowed difference. Defaults to 0.1.

        Returns:
            bool: True if fair (max diff <= threshold), False otherwise.
        """
        if not group_outcomes:
            return True

        values = list(group_outcomes.values())
        max_diff = max(values) - min(values)
        return max_diff <= threshold
