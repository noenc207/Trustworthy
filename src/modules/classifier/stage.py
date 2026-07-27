from src.modules.classifier.config import ClassifierConfig
from src.modules.classifier.exceptions import InvalidImageError
from src.modules.classifier.predictor import InferencePredictor
from src.modules.inference_engine.context import PipelineContext
from src.modules.inference_engine.stages import PipelineStage, StagePolicy


class LesionClassificationStage(PipelineStage):
    """Pipeline stage for lesion classification."""

    def __init__(self, config: ClassifierConfig, policy: StagePolicy | None = None):
        super().__init__(name="LesionClassificationStage", policy=policy)
        self.config = config
        self.predictor = None

    def initialize(self) -> None:
        self.predictor = InferencePredictor(self.config)

    def validate(self, context: PipelineContext) -> bool:
        # Check if preprocessing artifact exists
        # In M6.2, we stored it as context.preprocessing_result.
        # But we must transition to context.artifacts if implemented globally.
        # We'll support both for backward compatibility during the shift.
        if not hasattr(context, "artifacts"):
            context.artifacts = {}

        if hasattr(context, "preprocessing_result") and context.preprocessing_result is not None:
            return True

        if "preprocessing" in context.artifacts:
            return True

        context.add_error("No preprocessing result found.")
        return False

    def execute(self, context: PipelineContext) -> PipelineContext:
        # Resolve preprocessed image
        img = None
        if hasattr(context, "preprocessing_result") and context.preprocessing_result is not None:
            img = context.preprocessing_result.processed_image
        elif "preprocessing" in context.artifacts:
            img = context.artifacts["preprocessing"].processed_image

        if img is None:
            raise InvalidImageError("Processed image is None")

        # Execute Prediction
        pred, session = self.predictor.predict(
            image=img,
            request_id=context.request_id,
            execution_id=context.execution_id
        )

        # Store using the generic artifact container as requested
        if not hasattr(context, "artifacts"):
            context.artifacts = {}

        context.artifacts["classification"] = (pred, session)

        return context

    def cleanup(self) -> None:
        if self.predictor is not None:
            del self.predictor
            self.predictor = None
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
