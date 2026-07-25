from typing import Literal

from pydantic import BaseModel, Field


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
