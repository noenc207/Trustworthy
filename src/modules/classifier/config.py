
from pydantic import BaseModel, Field


class ClassifierConfig(BaseModel):
    model_name: str = "efficientnet_v2"
    model_version: str = "v1.0"
    backend: str = "torch"
    device: str = "cpu"
    class_names: list[str] = Field(default_factory=lambda: ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"])
    confidence_threshold: float = 0.5
    batch_size: int = 1
    input_size: tuple[int, int] = (224, 224)
    normalization: str = "standard"
    weights_path: str = "models/efficientnet_v2_v1.0.pt"
    lazy_loading: bool = True
