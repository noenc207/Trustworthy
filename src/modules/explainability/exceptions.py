class ExplainabilityError(Exception):
    """Base exception for Explainability module."""
    pass

class UnsupportedExplainerAlgorithmError(ExplainabilityError):
    pass

class ExplanationValidationError(ExplainabilityError):
    pass

class HookRegistrationError(ExplainabilityError):
    pass

class LayerResolutionError(ExplainabilityError):
    pass

class FaithfulnessEvaluationError(ExplainabilityError):
    pass

class StabilityEvaluationError(ExplainabilityError):
    pass

class SanityCheckFailedError(ExplainabilityError):
    pass

class MemoryLeakError(ExplainabilityError):
    pass

class ContextUnavailableError(ExplainabilityError):
    """Raised when OOD or calibration flags block XAI generation."""
    pass
