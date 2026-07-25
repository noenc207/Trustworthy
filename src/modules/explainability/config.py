from dataclasses import dataclass, field

from src.modules.explainability.enums import XAIAlgorithm


@dataclass
class ExplainabilityConfig:
    """Configuration for Clinical Explainability Engine."""

    # Core
    primary_algorithm: XAIAlgorithm = XAIAlgorithm.GRADCAM
    consensus_algorithms: list[XAIAlgorithm] = field(default_factory=lambda: [
        XAIAlgorithm.GRADCAM, XAIAlgorithm.GRADCAM_PP, XAIAlgorithm.SCORECAM
    ])
    target_layer: str | None = None
    fail_safe_conservative: bool = True

    # Context integration
    require_ood_clearance: bool = True
    require_calibration: bool = True

    # Evaluation Toggles
    enable_consensus: bool = True
    enable_faithfulness: bool = True
    enable_stability: bool = True
    enable_sanity_checks: bool = True

    # Stability Params
    noise_variance: float = 0.05
    rotation_degrees: int = 15
    blur_kernel_size: int = 3
    jpeg_quality: int = 50

    # Visual Params
    colormap: int = 2 # cv2.COLORMAP_JET
    opacity: float = 0.5
    interpolation: int = 1 # cv2.INTER_LINEAR
    output_dir: str = "outputs/explainability"
