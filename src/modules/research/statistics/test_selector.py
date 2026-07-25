"""Automatic statistical test selector."""

from typing import Any

import scipy.stats as stats

from .assumption_validator import AssumptionValidator


class TestSelector:
    """Selects and runs appropriate statistical tests."""

    @staticmethod
    def compare_two_groups(group1: list[float], group2: list[float], independent: bool = True) -> dict[str, Any]:
        """Compares two groups by automatically selecting the correct test.

        Args:
            group1 (List[float]): First group of data.
            group2 (List[float]): Second group of data.
            independent (bool, optional): Whether the groups are independent. Defaults to True.

        Returns:
            Dict[str, Any]: A dictionary containing the test name, statistic, and p-value.
        """
        is_normal1 = AssumptionValidator.validate_normality(group1)
        is_normal2 = AssumptionValidator.validate_normality(group2)
        is_normal = is_normal1 and is_normal2

        stat = 0.0
        p = 1.0

        if independent:
            if is_normal:
                is_equal_var = AssumptionValidator.validate_variance(group1, group2)
                if is_equal_var:
                    # Independent t-test
                    stat, p = stats.ttest_ind(group1, group2)
                    test_name = "Independent t-test"
                else:
                    # Welch's t-test
                    stat, p = stats.ttest_ind(group1, group2, equal_var=False)
                    test_name = "Welch's t-test"
            else:
                # Mann-Whitney U test
                stat, p = stats.mannwhitneyu(group1, group2)
                test_name = "Mann-Whitney U test"
        else:
            if is_normal:
                # Paired t-test
                stat, p = stats.ttest_rel(group1, group2)
                test_name = "Paired t-test"
            else:
                # Wilcoxon signed-rank test
                res = stats.wilcoxon(group1, group2)
                stat, p = res.statistic, res.pvalue
                test_name = "Wilcoxon signed-rank test"

        return {
            "test_name": test_name,
            "statistic": float(stat),
            "p_value": float(p),
        }
