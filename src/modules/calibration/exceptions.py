class CalibrationException(Exception):
    """Base exception for Calibration module."""
    pass

class UnsupportedCalibrationAlgorithmError(CalibrationException):
    pass

class CalibrationEvaluationError(CalibrationException):
    pass

class MissingPredictionArtifactError(CalibrationException):
    pass
