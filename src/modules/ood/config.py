from pydantic import BaseModel, Field

class OODConfig(BaseModel):
    algorithm: str = "msp"
    threshold: float = 0.5
    energy_temperature: float = 1.0
    enable_mahalanobis: bool = False
    fail_safe_conservative: bool = True
