from dataclasses import dataclass, field

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
