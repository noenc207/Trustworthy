import os
from pathlib import Path

files = {}

# 1. config.py
files["src/modules/preprocessing/config.py"] = """
from pydantic import BaseModel, Field
from typing import Literal

class CLAHEConfig(BaseModel):
    enabled: bool = False
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)

class HairRemovalConfig(BaseModel):
    enabled: bool = False
    kernel_size: int = 15
    threshold: int = 10
    inpaint_radius: int = 3

class NormalizationConfig(BaseModel):
    enabled: bool = True
    mode: Literal["minmax", "standard"] = "standard"
    mean: list[float] = [0.485, 0.456, 0.406]
    std: list[float] = [0.229, 0.224, 0.225]

class ArtifactConfig(BaseModel):
    enabled: bool = False
    method: str = "inpaint"
    kernel_size: int = 5
    strength: float = 1.0

class ROIConfig(BaseModel):
    enabled: bool = False
    strategy: Literal["center", "otsu"] = "center"

class ResizeConfig(BaseModel):
    enabled: bool = True
    target_size: tuple[int, int] = (224, 224)
    maintain_aspect_ratio: bool = True

class PreprocessingConfig(BaseModel):
    random_seed: int = 42
    resize: ResizeConfig = Field(default_factory=ResizeConfig)
    clahe: CLAHEConfig = Field(default_factory=CLAHEConfig)
    hair_removal: HairRemovalConfig = Field(default_factory=HairRemovalConfig)
    artifact_suppression: ArtifactConfig = Field(default_factory=ArtifactConfig)
    roi: ROIConfig = Field(default_factory=ROIConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)
"""

# 2. results.py
files["src/modules/preprocessing/results.py"] = """
from dataclasses import dataclass, field
from typing import Any
import numpy as np

@dataclass
class PreprocessingResult:
    processed_image: np.ndarray
    original_shape: tuple[int, int]
    output_shape: tuple[int, int]
    applied_operations: list[str] = field(default_factory=list)
    execution_time: float = 0.0
    preprocessing_version: str = "1.0"
    warnings: list[str] = field(default_factory=list)
"""

# 3. operations/base.py
files["src/modules/preprocessing/operations/base.py"] = """
from abc import ABC, abstractmethod
import numpy as np

class AbstractPreprocessingOperation(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def is_enabled(self) -> bool:
        pass
"""

# 4. operations/resize.py
files["src/modules/preprocessing/operations/resize.py"] = """
import cv2
import numpy as np
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation
from src.modules.preprocessing.config import ResizeConfig

class ResizeOperation(AbstractPreprocessingOperation):
    def __init__(self, config: ResizeConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Resize"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image
        
        target_w, target_h = self.config.target_size
        h, w = image.shape[:2]
        
        if not self.config.maintain_aspect_ratio:
            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
            
        # Maintain aspect ratio with padding
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        delta_w = target_w - new_w
        delta_h = target_h - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        color = [0, 0, 0]
        if len(image.shape) == 2:
            color = [0]
            
        new_im = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return new_im
"""

# 5. operations/clahe.py
files["src/modules/preprocessing/operations/clahe.py"] = """
import cv2
import numpy as np
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation
from src.modules.preprocessing.config import CLAHEConfig

class CLAHEOperation(AbstractPreprocessingOperation):
    def __init__(self, config: CLAHEConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "CLAHE"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image
            
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit, 
            tileGridSize=self.config.tile_grid_size
        )
        
        if len(image.shape) == 2:
            return clahe.apply(image)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        else:
            return image
"""

# 6. operations/hair_removal.py
files["src/modules/preprocessing/operations/hair_removal.py"] = """
import cv2
import numpy as np
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation
from src.modules.preprocessing.config import HairRemovalConfig

class HairRemovalOperation(AbstractPreprocessingOperation):
    def __init__(self, config: HairRemovalConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "HairRemoval"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image
            
        # DullRazor algorithm
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
            
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (self.config.kernel_size, self.config.kernel_size))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, mask = cv2.threshold(blackhat, self.config.threshold, 255, cv2.THRESH_BINARY)
        
        inpainted = cv2.inpaint(image, mask, self.config.inpaint_radius, cv2.INPAINT_TELEA)
        return inpainted
"""

# 7. operations/normalize.py
files["src/modules/preprocessing/operations/normalize.py"] = """
import numpy as np
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation
from src.modules.preprocessing.config import NormalizationConfig

class NormalizeOperation(AbstractPreprocessingOperation):
    def __init__(self, config: NormalizationConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "Normalize"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image
            
        img_float = image.astype(np.float32)
        
        if self.config.mode == "minmax":
            min_val = img_float.min()
            max_val = img_float.max()
            if max_val > min_val:
                img_float = (img_float - min_val) / (max_val - min_val)
        elif self.config.mode == "standard":
            if img_float.max() > 1.0:
                img_float = img_float / 255.0
                
            mean = np.array(self.config.mean, dtype=np.float32)
            std = np.array(self.config.std, dtype=np.float32)
            
            img_float = (img_float - mean) / std
            
        return img_float
"""

# 8. operations/artifact.py
files["src/modules/preprocessing/operations/artifact.py"] = """
import cv2
import numpy as np
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation
from src.modules.preprocessing.config import ArtifactConfig

class ArtifactSuppressionOperation(AbstractPreprocessingOperation):
    def __init__(self, config: ArtifactConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "ArtifactSuppression"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image
        return image
"""

# 9. operations/roi.py
files["src/modules/preprocessing/operations/roi.py"] = """
import cv2
import numpy as np
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation
from src.modules.preprocessing.config import ROIConfig

class ROIOperation(AbstractPreprocessingOperation):
    def __init__(self, config: ROIConfig):
        self.config = config

    @property
    def name(self) -> str:
        return "ROI Extraction"

    def is_enabled(self) -> bool:
        return self.config.enabled

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if not self.is_enabled():
            return image
        if self.config.strategy == "center":
            h, w = image.shape[:2]
            crop_size = min(h, w)
            start_y = (h - crop_size) // 2
            start_x = (w - crop_size) // 2
            return image[start_y:start_y+crop_size, start_x:start_x+crop_size]
        return image
"""

# 10. operations/__init__.py
files["src/modules/preprocessing/operations/__init__.py"] = """
from src.modules.preprocessing.operations.base import AbstractPreprocessingOperation
from src.modules.preprocessing.operations.resize import ResizeOperation
from src.modules.preprocessing.operations.clahe import CLAHEOperation
from src.modules.preprocessing.operations.hair_removal import HairRemovalOperation
from src.modules.preprocessing.operations.normalize import NormalizeOperation
from src.modules.preprocessing.operations.artifact import ArtifactSuppressionOperation
from src.modules.preprocessing.operations.roi import ROIOperation

__all__ = [
    "AbstractPreprocessingOperation",
    "ResizeOperation",
    "CLAHEOperation",
    "HairRemovalOperation",
    "NormalizeOperation",
    "ArtifactSuppressionOperation",
    "ROIOperation"
]
"""

# 11. stage.py
files["src/modules/preprocessing/stage.py"] = """
import time
import numpy as np
import cv2
from PIL import Image
from typing import Any

from src.modules.inference_engine.stages import PipelineStage, StagePolicy
from src.modules.inference_engine.context import PipelineContext
from src.modules.preprocessing.config import PreprocessingConfig
from src.modules.preprocessing.results import PreprocessingResult
from src.modules.preprocessing.operations import (
    AbstractPreprocessingOperation,
    ResizeOperation,
    CLAHEOperation,
    HairRemovalOperation,
    NormalizeOperation,
    ArtifactSuppressionOperation,
    ROIOperation
)

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
        pass
"""

def write_files():
    base_dir = Path("d:/Trustworthy")
    for file_path, content in files.items():
        full_path = base_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content.lstrip(), encoding="utf-8")
        print(f"Created: {file_path}")

if __name__ == "__main__":
    write_files()
