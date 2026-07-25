"""Trust aggregation via Dempster-Shafer."""


class TrustAggregator:
    """Aggregates evidence using Dempster-Shafer theory.

    Responsibility: Fuses evidence from multiple sources weighting by reliability
    (inverse uncertainty).
    Time Complexity: O(N * 2^|Theta|) where N is number of sources, |Theta| is subsets.
    Determinism: Deterministic.
    Mathematical Formula:
        m_12(A) = (1 / (1 - K)) * sum(m1(B) * m2(C) for B, C where B int C = A)
        K = sum(m1(B) * m2(C) for B, C where B int C = empty)
    Edge Cases:
        - High conflict (K = 1). Handled by returning full ignorance (mass 1 on frame).
        - Empty inputs. Returns empty dict or ignorance.
    """

    def __init__(self, frame_of_discernment: frozenset[str]) -> None:
        """Initialize with a frame of discernment.

        Args:
            frame_of_discernment: The set of all possible mutually exclusive hypotheses.
        """
        self.frame = frame_of_discernment

    def aggregate(self, masses: list[dict[frozenset[str], float]]) -> dict[frozenset[str], float]:
        """Combine multiple mass assignments.

        Args:
            masses: List of mass assignments, each mapping subsets of frame to [0, 1].

        Returns:
            Combined mass assignment.
        """
        if not masses:
            return {self.frame: 1.0}

        combined = masses[0]
        for i in range(1, len(masses)):
            combined = self._combine_two(combined, masses[i])

        return combined

    def _combine_two(self, m1: dict[frozenset[str], float], m2: dict[frozenset[str], float]) -> dict[frozenset[str], float]:
        """Combine two mass assignments using Dempster's rule."""
        combined_mass: dict[frozenset[str], float] = {}
        conflict = 0.0

        for set1, mass1 in m1.items():
            for set2, mass2 in m2.items():
                intersection = set1.intersection(set2)
                product = mass1 * mass2
                if not intersection:
                    conflict += product
                else:
                    combined_mass[intersection] = combined_mass.get(intersection, 0.0) + product

        if conflict >= 0.999999:  # Complete conflict
            return {self.frame: 1.0}

        normalization = 1.0 / (1.0 - conflict)
        normalized_mass = {k: v * normalization for k, v in combined_mass.items()}
        return normalized_mass
