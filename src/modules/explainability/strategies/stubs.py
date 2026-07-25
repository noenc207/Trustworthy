from src.modules.explainability.strategies.gradcam_wrappers import (
    LayerCAMStrategy, EigenCAMStrategy, AblationCAMStrategy
)
from src.modules.explainability.strategies.captum_wrappers import (
    GuidedBackpropStrategy, GuidedGradCAMStrategy, IntegratedGradientsStrategy, DeepLIFTStrategy
)
from src.modules.explainability.strategies.captum_wrappers import CaptumBaseStrategy
from typing import Any

class LRPStrategy(CaptumBaseStrategy):
    def collect(self, model: Any, input_tensor: Any, target_class: int, hook_manager: Any) -> None:
        super().collect(model, input_tensor, target_class, hook_manager)
        from captum.attr import LRP
        self.captum_method = LRP(model)

