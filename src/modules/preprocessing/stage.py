import time

import cv2
import numpy as np
from PIL import Image

from src.modules.inference_engine.context import PipelineContext
from src.modules.inference_engine.stages import PipelineStage, StagePolicy
from src.modules.preprocessing.config import PreprocessingConfig
from src.modules.preprocessing.operations import (
    AbstractPreprocessingOperation,
    ArtifactSuppressionOperation,
    CLAHEOperation,
    HairRemovalOperation,
    NormalizeOperation,
    ResizeOperation,
    ROIOperation,
)
from src.modules.preprocessing.results import PreprocessingResult


class ImagePreprocessor(PipelineStage):
    def __init__(self, config: PreprocessingConfig, policy: StagePolicy | None = None):
        super().__init__(name="ImagePreprocessor", policy=policy)
        self.config = config
        self.operations: list[AbstractPreprocessingOperation] = []

    def initialize(self) -> None:
        np.random.seed(self.config.random_seed)

        # Build operation pipeline (Strategy Pattern)
        self.operations = [
            ROIOperation(self.config.roi),
            HairRemovalOperation(self.config.hair_removal),
            CLAHEOperation(self.config.clahe),
            ArtifactSuppressionOperation(self.config.artifact_suppression),
            ResizeOperation(self.config.resize),
            NormalizeOperation(self.config.normalization)
        ]

    def validate(self, context: PipelineContext) -> bool:
        if context.raw_image is None:
            context.add_error("raw_image is None.")
            return False
        return True

    def execute(self, context: PipelineContext) -> PipelineContext:
        start_time = time.time()

        # Handle input types safely
        img: np.ndarray
        if isinstance(context.raw_image, Image.Image):
            img = np.array(context.raw_image)
            if len(img.shape) == 3 and img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif isinstance(context.raw_image, np.ndarray):
            img = context.raw_image.copy()
            if len(img.shape) == 3 and img.shape[-1] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            raise ValueError(f"Unsupported image format: {type(context.raw_image)}")

        if img.size == 0:
            raise ValueError("Empty image tensor provided.")

        original_shape = img.shape[:2]
        applied_ops = []

        for op in self.operations:
            if op.is_enabled():
                img = op(img)
                applied_ops.append(op.name)

        output_shape = img.shape[:2]
        exec_time = time.time() - start_time

        result = PreprocessingResult(
            processed_image=img,
            original_shape=original_shape,
            output_shape=output_shape,
            applied_operations=applied_ops,
            execution_time=exec_time
        )

        # Bind safely to context
        context.preprocessing_result = result
        return context

    def cleanup(self) -> None:
        """No resources to cleanup."""
