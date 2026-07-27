from src.modules.inference_engine.context import PipelineContext
from src.modules.inference_engine.stages import PipelineStage, StagePolicy
from src.modules.ood.config import OODConfig
from src.modules.ood.detector import DefaultOODDetector
from src.modules.ood.exceptions import MissingClassificationArtifactError


class OODDetectionStage(PipelineStage):
    """Pipeline stage for Out-of-Distribution evaluation."""

    def __init__(self, config: OODConfig, policy: StagePolicy | None = None):
        super().__init__(name="OODDetectionStage", policy=policy)
        self.config = config
        self.detector = None

    def initialize(self) -> None:
        self.detector = DefaultOODDetector(self.config)

    def validate(self, context: PipelineContext) -> bool:
        if not hasattr(context, "artifacts") or "classification" not in context.artifacts:
            context.add_error("No classification result found in artifacts.")
            return False
        return True

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not self.validate(context):
            raise MissingClassificationArtifactError("Missing classification artifact.")

        pred, session = context.artifacts["classification"]

        # Raw logits not typically saved in DTO, fallback to probs
        ood_result = self.detector.evaluate(pred, raw_logits=None)

        context.artifacts["ood"] = ood_result

        return context

    def cleanup(self) -> None:
        """Release OOD detector resources."""
        if hasattr(self, 'detector'):
            del self.detector
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
