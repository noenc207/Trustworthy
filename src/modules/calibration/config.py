from pathlib import Path

from pydantic import BaseModel, Field


class CalibrationConfig(BaseModel):
    algorithm: str = "temperature"

    # Optimizer settings for Temperature Scaling
    optimizer: str = "lbfgs"  # 'lbfgs' or 'adam'
    learning_rate: float = 0.01
    max_iterations: int = 1000
    patience: int = 10
    min_loss_improvement: float = 1e-4
    gradient_tolerance: float = 1e-5

    # Visualization settings
    generate_diagrams: bool = True
    output_dir: Path = Field(default=Path("outputs/calibration"))

    # Fallback and failsafe settings
    fail_safe_conservative: bool = True

    # Bin configuration for metrics
    num_bins: int = 15  # Configurable: 10, 15, 20, 30
