"""Statistical assumption validator."""


import scipy.stats as stats


class AssumptionValidator:
    """Validates assumptions for statistical tests."""

    @staticmethod
    def validate_normality(data: list[float], alpha: float = 0.05) -> bool:
        """Validates normality assumption using Shapiro-Wilk test.

        Args:
            data (List[float]): The data to test.
            alpha (float, optional): The significance level. Defaults to 0.05.

        Returns:
            bool: True if normal (fail to reject null), False otherwise.
        """
        if len(data) < 3:
            return False

        _, p_value = stats.shapiro(data)
        return bool(p_value >= alpha)

    @staticmethod
    def validate_variance(
        group1: list[float], group2: list[float], alpha: float = 0.05
    ) -> bool:
        """Validates equal variance assumption using Levene test.

        Args:
            group1 (List[float]): The first group's data.
            group2 (List[float]): The second group's data.
            alpha (float, optional): The significance level. Defaults to 0.05.

        Returns:
            bool: True if variances are equal (fail to reject null), False otherwise.
        """
        if len(group1) < 2 or len(group2) < 2:
            return False

        _, p_value = stats.levene(group1, group2)
        return bool(p_value >= alpha)
