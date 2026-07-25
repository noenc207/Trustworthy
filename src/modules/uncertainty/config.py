from pydantic import BaseModel, Field

class UncertaintyConfig(BaseModel):
    algorithm: str = "entropy"
    threshold: float = 0.5
    sampling_count: int = 30
    confidence_interval_level: float = 0.95
    fail_safe_conservative: bool = True
