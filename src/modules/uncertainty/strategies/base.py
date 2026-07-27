from abc import ABC

from src.modules.uncertainty.interfaces import UncertaintyStrategy


class AbstractUncertaintyStrategy(UncertaintyStrategy, ABC):
    """Base class for Uncertainty strategies."""
    pass
