from src.modules.inference_engine.context import PipelineContext
from src.modules.inference_engine.stages import PipelineStage, StagePolicy
from src.modules.uncertainty.config import UncertaintyConfig
from src.modules.uncertainty.estimator import DefaultUncertaintyEstimator
from src.modules.uncertainty.exceptions import MissingArtifactError


class UncertaintyEstimationStage(PipelineStage):
    """Pipeline stage for Uncertainty evaluation."""

    def __init__(self, config: UncertaintyConfig, policy: StagePolicy | None = None):
        super().__init__(name="UncertaintyEstimationStage", policy=policy)
        self.config = config
        self.estimator = None

    def initialize(self) -> None:
        self.estimator = DefaultUncertaintyEstimator(self.config)

    def validate(self, context: PipelineContext) -> bool:
        if not hasattr(context, "artifacts") or "classification" not in context.artifacts:
            context.add_error("No classification result found in artifacts.")
            return False
        return True

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not self.validate(context):
            raise MissingArtifactError("Missing classification artifact.")

        pred, session = context.artifacts["classification"]
        ood_res = context.artifacts.get("ood")

        # In a real environment with MC Dropout, we might trigger the classifier here
        # to generate stochastic samples if supported. Since this module DOES NOT
        # classify, we assume samples are either passed in context.artifacts
        # or we rely on single-pass metrics if samples are None.
        stochastic_samples = context.artifacts.get("stochastic_samples")

        uncertainty_res = self.estimator.evaluate(
            classification_result=pred,
            ood_result=ood_res,
            raw_logits=None,
            stochastic_samples=stochastic_samples
        )

        context.artifacts["uncertainty"] = uncertainty_res

        return context

    def cleanup(self) -> None:
        """Release uncertainty estimator resources."""
        if hasattr(self, 'estimator'):
            del self.estimator
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
