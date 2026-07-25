from src.modules.classifier.exceptions import ModelNotLoadedError
from src.modules.explainability.config import ExplainabilityConfig
from src.modules.explainability.exceptions import ContextUnavailableError
from src.modules.explainability.orchestrator import ExplainabilityEngine


class ExplainabilityStage:
    """Pipeline Stage for Clinical Explainability."""

    def __init__(self, config: ExplainabilityConfig, backend_adapter):
        self.config = config
        self.engine = ExplainabilityEngine(config, backend_adapter)

    def initialize(self) -> None:
        import logging
        logging.getLogger(__name__).info("Initializing ExplainabilityStage")

    def validate(self, context) -> bool:
        if "classification" not in context.artifacts:
            return False
        if "preprocessed_image" not in context.artifacts:
            return False
        return "input_tensor" in context.artifacts

    def execute(self, context):
        pred_res, model = context.artifacts["classification"]
        if model is None:
            raise ModelNotLoadedError("Model handle is missing from classification artifact.")

        # OOD Integration
        if self.config.require_ood_clearance:
            ood_result = context.artifacts.get("ood")
            if ood_result and ood_result.is_ood:
                raise ContextUnavailableError("Explanation unavailable: Image is Out-of-Distribution.")

        # Calibration Integration
        if self.config.require_calibration:
            context.artifacts.get("calibration")
            # For this stub, we just pretend we read it. In reality, we might lower trust.

        input_tensor = context.artifacts["input_tensor"]
        original_image = context.artifacts["preprocessed_image"]

        xai_res = self.engine.evaluate(
            model=model,
            image=original_image,
            tensor=input_tensor,
            prediction=pred_res
        )

        context.artifacts["explainability"] = xai_res
        return context
