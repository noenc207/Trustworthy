from typing import Any
from src.modules.inference_engine.stages import PipelineStage, StagePolicy
from src.modules.inference_engine.context import PipelineContext
from src.modules.calibration.config import CalibrationConfig
from src.modules.calibration.calibrator import DefaultCalibrationEngine
from src.modules.calibration.exceptions import MissingPredictionArtifactError

class CalibrationStage(PipelineStage):
    """Pipeline stage for Confidence Calibration."""
    
    def __init__(self, config: CalibrationConfig, policy: StagePolicy | None = None):
        super().__init__(name="CalibrationStage", policy=policy)
        self.config = config
        self.engine = None

    def initialize(self) -> None:
        self.engine = DefaultCalibrationEngine(self.config)

    def validate(self, context: PipelineContext) -> bool:
        if not hasattr(context, "artifacts") or "classification" not in context.artifacts:
            context.add_error("No classification result found for calibration.")
            return False
        return True

    def execute(self, context: PipelineContext) -> PipelineContext:
        if not self.validate(context):
            raise MissingPredictionArtifactError("Missing classification artifact.")
            
        pred, session = context.artifacts["classification"]
        
        raw_logits = context.artifacts.get("logits")
        true_labels = context.artifacts.get("true_labels")
        
        calib_res = self.engine.evaluate(
            classification_result=pred,
            raw_logits=raw_logits,
            true_labels=true_labels
        )
        
        context.artifacts["calibration"] = calib_res
        
        return context

    def cleanup(self) -> None:
        pass
