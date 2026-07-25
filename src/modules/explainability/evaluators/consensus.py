
import numpy as np

from src.modules.explainability.interfaces import ConsensusEngineInterface
from src.modules.explainability.result import ConsensusResult


class MultiXAIConsensus(ConsensusEngineInterface):
    """Blends multiple XAI algorithms into a consensus map and calculates agreement."""

    def aggregate(self, heatmaps: list[np.ndarray], algorithms: list[str]) -> ConsensusResult:
        if not heatmaps:
            return ConsensusResult(0.0, None, None, [])

        stacked = np.stack(heatmaps, axis=0) # (N, H, W)

        # Mean consensus heatmap
        consensus_hm = np.mean(stacked, axis=0)

        # Variance as disagreement
        disagreement_map = np.var(stacked, axis=0)

        # Agreement score: higher variance across maps = lower agreement
        mean_var = np.mean(disagreement_map)
        # Normalize: var is in [0, 1] assuming heatmaps in [0, 1]
        agreement_score = max(0.0, 100.0 - (mean_var * 1000.0)) # heuristic scaling
        agreement_score = min(100.0, agreement_score)

        return ConsensusResult(
            agreement_score=agreement_score,
            disagreement_map=disagreement_map,
            consensus_heatmap=consensus_hm,
            algorithms_used=algorithms
        )
