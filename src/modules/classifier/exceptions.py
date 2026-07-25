class ClassifierDomainError(Exception):
    def __init__(self, message: str, error_code: str, context: dict = None, recommended_action: str = ""):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.recommended_action = recommended_action

class CheckpointError(ClassifierDomainError): pass
class ModelInitializationError(ClassifierDomainError): pass
class PredictionError(ClassifierDomainError): pass
class InvalidInputError(ClassifierDomainError): pass
class DeviceMismatchError(ClassifierDomainError): pass
class ConfigurationError(ClassifierDomainError): pass
class NumericalInstabilityError(ClassifierDomainError): pass
class SerializationError(ClassifierDomainError): pass

# Legacy compatibility
class UnsupportedBackboneError(ClassifierDomainError): pass
class UnsupportedBackendError(ClassifierDomainError): pass
class InvalidImageError(ClassifierDomainError): pass
