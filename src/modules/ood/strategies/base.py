from abc import ABC

from src.modules.ood.interfaces import OODStrategy


class AbstractOODStrategy(OODStrategy, ABC):
    """Base class for OOD strategies."""
    pass
