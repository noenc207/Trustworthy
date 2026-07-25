class OODException(Exception):
    """Base exception for OOD module."""
    pass

class UnsupportedOODAlgorithmError(OODException):
    pass

class OODEvaluationError(OODException):
    pass

class MissingClassificationArtifactError(OODException):
    pass
