import hashlib
import time
import traceback
from typing import Any

import numpy as np

from src.modules.classifier.interfaces import BackendAdapter
from src.modules.classifier.result import PredictionResult
from src.modules.explainability.cache import ExplainabilityCache
from src.modules.explainability.config import ExplainabilityConfig
from src.modules.explainability.exceptions import UnsupportedExplainerAlgorithmError
from src.modules.explainability.hook_manager import HookManager
from src.modules.explainability.interfaces import ExplainerStrategy
from src.modules.explainability.layer_resolver import LayerResolver
from src.modules.explainability.report import ReportGenerator
from src.modules.explainability.result import ExplainabilityResult
from src.modules.explainability.visualization import ExplainabilityVisualization


class ExplainerEngine:
    """Orchestrator for the Explainability module."""

    def __init__(self, config: ExplainabilityConfig, backend_adapter: BackendAdapter):
        self.config = config
        self.adapter = backend_adapter
        self.cache = ExplainabilityCache()

    def _load_strategy(self) -> ExplainerStrategy:
        alg = self.config.algorithm.lower()
        if alg == "gradcam":
            from src.modules.explainability.strategies.gradcam import GradCAMStrategy
            return GradCAMStrategy(self.config)
        elif alg == "gradcam++":
            from src.modules.explainability.strategies.gradcampp import GradCAMPPStrategy
            return GradCAMPPStrategy(self.config)
        elif alg == "scorecam":
            from src.modules.explainability.strategies.scorecam import ScoreCAMStrategy
            return ScoreCAMStrategy(self.config)
        else:
            raise UnsupportedExplainerAlgorithmError(f"Unsupported XAI algorithm: {self.config.algorithm}")

    def _hash_image(self, img: np.ndarray) -> str:
        # A lightweight hash for caching purposes
        sub = img[::10, ::10].copy()
        return hashlib.md5(sub.tobytes()).hexdigest()

    def evaluate(
        self,
        model: Any,
        input_tensor: Any,
        original_image: np.ndarray,
        classification_result: PredictionResult
    ) -> ExplainabilityResult:

        start_time = time.time()
        warnings = []
        target_class = classification_result.predicted_index

        # 1. Cache Check
        img_hash = self._hash_image(original_image)
        cached = self.cache.get(model, target_class, self.config, img_hash)
        if cached:
            return cached

        # 2. Layer Resolution
        try:
            layer_name = self.config.target_layer or LayerResolver.resolve(model)
        except Exception as e:
            layer_name = "unknown"
            warnings.append(str(e))
            if not self.config.fail_safe_conservative:
                raise e

        hook_manager = HookManager(self.adapter, model)

        try:
            strategy = self._load_strategy()

            # 3. Hook Registration
            if layer_name != "unknown":
                hook_manager.register_hooks(layer_name)

            # 4. Collection (Forward/Backward)
            strategy.collect(model, input_tensor, target_class, hook_manager)

            # 5. Computation & Normalization
            raw = strategy.compute(hook_manager.get_activations(), hook_manager.get_gradients())
            norm = strategy.normalize(raw)

            # 6. Statistics & Validation
            stats = strategy.statistics(norm)
            strategy.validate(norm, stats)

            # 7. Rendering
            overlay = strategy.render(norm, original_image)

            # 8. Output Visualizations
            ExplainabilityVisualization.generate_all(
                self.config.output_dir, raw, norm, overlay, original_image
            )

            exec_time = time.time() - start_time

            res = ExplainabilityResult(
                algorithm=self.config.algorithm,
                layer_name=layer_name,
                heatmap=norm,
                overlay_image=overlay,
                importance_score=stats.get("peak_activation", 0.0),
                confidence_region=stats.get("bounding_box", None),
                activation_statistics=strategy.report(),
                inference_time=exec_time,
                warnings=warnings,
                **{k: v for k, v in stats.items() if k in ExplainabilityResult.__dataclass_fields__ and k not in ['bounding_box']}
            )

            # 9. Reporting
            ReportGenerator.generate(res, self.config.output_dir)

            # 10. Cache Update
            self.cache.set(model, target_class, self.config, img_hash, res)

            return res

        except Exception as e:
            warnings.append(f"Explainability execution failed: {e}\n{traceback.format_exc()}")
            if not self.config.fail_safe_conservative:
                hook_manager.cleanup()
                raise e

            return ExplainabilityResult(
                algorithm=self.config.algorithm,
                layer_name=layer_name,
                heatmap=None,
                overlay_image=None,
                importance_score=0.0,
                confidence_region=None,
                activation_statistics={},
                inference_time=time.time() - start_time,
                warnings=warnings
            )
        finally:
            hook_manager.cleanup()
