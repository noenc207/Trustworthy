class UncertaintyException(Exception):
    """Base exception for Uncertainty module."""
    pass

class UnsupportedUncertaintyAlgorithmError(UncertaintyException):
    pass

class UncertaintyEvaluationError(UncertaintyException):
    pass

class MissingArtifactError(UncertaintyException):
    pass
