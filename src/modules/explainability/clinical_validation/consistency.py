
import numpy as np


class EvidenceConsistencyMatrix:
    def __init__(self):
        pass

    def build_matrix(self, evidence_scores: list[float]) -> np.ndarray:
        n = len(evidence_scores)
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                matrix[i, j] = 1.0 - abs(evidence_scores[i] - evidence_scores[j])
        return matrix
