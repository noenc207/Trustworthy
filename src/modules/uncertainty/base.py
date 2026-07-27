from abc import ABC, abstractmethod

import numpy as np


class BaseUncertaintyEstimator(ABC):
    @abstractmethod
    def estimate(self, probabilities: np.ndarray) -> dict:
        pass
