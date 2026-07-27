from abc import ABC

from src.modules.calibration.interfaces import CalibrationStrategy


class AbstractCalibrationStrategy(CalibrationStrategy, ABC):
    """Base class for Calibration strategies."""
    pass
