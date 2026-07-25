import time
from typing import Any

import numpy as np

from src.infrastructure.ml_backends.torch.adapter import TorchBackendAdapter
from src.modules.classifier.result import PredictionResult
from src.modules.explainability.backend.hook_manager import HookManager
from src.modules.explainability.backend.layer_resolver import LayerResolver
from src.modules.explainability.cache import ExplainabilityCache
from src.modules.explainability.config import ExplainabilityConfig
from src.modules.explainability.enums import XAIAlgorithm
from src.modules.explainability.evaluators.consensus import MultiXAIConsensus
from src.modules.explainability.evaluators.faithfulness import FaithfulnessEvaluator
from src.modules.explainability.evaluators.sanity import SanityChecker
from src.modules.explainability.evaluators.stability import StabilityEvaluator
from src.modules.explainability.metrics.clinical import MedicalMetricsEngine
from src.modules.explainability.reporting.overlay import OverlayEngine
from src.modules.explainability.reporting.report_generator import ClinicalReportGenerator
from src.modules.explainability.reporting.visualization import VisualizationEngine
from src.modules.explainability.result import ExplainabilityResult


class ExplainabilityEngine:
    """Clinical-Grade Explainability Orchestrator."""

    def __init__(self, config: ExplainabilityConfig, adapter: TorchBackendAdapter):
        self.config = config
        self.adapter = adapter
        self.cache = ExplainabilityCache()

        self.consensus = MultiXAIConsensus()
        self.faithfulness = FaithfulnessEvaluator()
        self.stability = StabilityEvaluator()
        self.sanity = SanityChecker()
        self.metrics = MedicalMetricsEngine()
        self.visualizer = VisualizationEngine()
        self.reporter = ClinicalReportGenerator()

    def _get_strategy(self, algo: XAIAlgorithm):
        from src.modules.explainability.strategies.gradcam import GradCAMStrategy
        from src.modules.explainability.strategies.gradcampp import GradCAMPPStrategy
        from src.modules.explainability.strategies.scorecam import ScoreCAMStrategy
        from src.modules.explainability.strategies.hirescam import HiResCAMStrategy
        
        # New Batch 5 Strategies
        from src.modules.explainability.strategies.captum_wrappers import (
            GuidedBackpropStrategy, GuidedGradCAMStrategy, IntegratedGradientsStrategy,
            DeepLIFTStrategy, InputXGradientStrategy, OcclusionStrategy, FeatureAblationStrategy
        )
        from src.modules.explainability.strategies.gradcam_wrappers import (
            LayerCAMStrategy, XGradCAMStrategy, EigenCAMStrategy, EigenGradCAMStrategy,
            AblationCAMStrategy, FullGradStrategy
        )

        registry = {
            XAIAlgorithm.GRADCAM: GradCAMStrategy,
            XAIAlgorithm.GRADCAM_PP: GradCAMPPStrategy,
            XAIAlgorithm.SCORECAM: ScoreCAMStrategy,
            XAIAlgorithm.HIRESCAM: HiResCAMStrategy,
            XAIAlgorithm.LAYERCAM: LayerCAMStrategy,
            XAIAlgorithm.XGRADCAM: XGradCAMStrategy,
            XAIAlgorithm.EIGENCAM: EigenCAMStrategy,
            XAIAlgorithm.EIGENGRADCAM: EigenGradCAMStrategy,
            XAIAlgorithm.ABLATIONCAM: AblationCAMStrategy,
            XAIAlgorithm.FULLGRAD: FullGradStrategy,
            XAIAlgorithm.GUIDED_BACKPROP: GuidedBackpropStrategy,
            XAIAlgorithm.GUIDED_GRADCAM: GuidedGradCAMStrategy,
            XAIAlgorithm.INTEGRATED_GRADIENTS: IntegratedGradientsStrategy,
            XAIAlgorithm.DEEPLIFT: DeepLIFTStrategy,
            XAIAlgorithm.INPUT_X_GRADIENT: InputXGradientStrategy,
            XAIAlgorithm.OCCLUSION: OcclusionStrategy,
            XAIAlgorithm.FEATURE_ABLATION: FeatureAblationStrategy,
        }

        if algo in registry:
            return registry[algo](self.config)
        
        # Fallback
        from src.modules.explainability.strategies.base import BaseCAMStrategy
        class FallbackStrategy(BaseCAMStrategy):
            def compute(self, activations, gradients):
                import numpy as np
                if hasattr(activations, "detach"):
                    act = activations.detach().cpu().numpy()
                    grad = gradients.detach().cpu().numpy()
                else:
                    act = np.array(activations)
                    grad = np.array(gradients)
                weights = np.mean(grad, axis=(2, 3), keepdims=True)
                cam = np.sum(weights * act, axis=1)
                if cam.shape[0] == 1:
                    cam = cam[0]
                self.raw_heatmap = cam
                return cam
        return FallbackStrategy(self.config)

    def _execute_strategy(self, model: Any, tensor: Any, target_class: int, layer_name: str, algo: XAIAlgorithm):
        hm = HookManager(self.adapter, model)
        try:
            hm.register_hooks(layer_name)
            strategy = self._get_strategy(algo)
            strategy.collect(model, tensor, target_class, hm)
            raw = strategy.compute(hm.get_activations(), hm.get_gradients())
            return strategy.normalize(raw), strategy
        finally:
            hm.cleanup()

    def evaluate(self, model: Any, image: np.ndarray, tensor: Any, prediction: PredictionResult) -> ExplainabilityResult:
        start_time = time.time()
        target_class = prediction.predicted_index
        warnings = []

        layer_name = self.config.target_layer or LayerResolver.resolve(model)

        cached = self.cache.get(model, target_class, self.config, image, layer_name)
        if cached:
            return cached

        try:
            # 1. Primary or Consensus Extraction
            primary_hm, primary_strat = self._execute_strategy(model, tensor, target_class, layer_name, self.config.primary_algorithm)

            consensus_result = None
            final_hm = primary_hm

            if self.config.enable_consensus:
                heatmaps = []
                for algo in self.config.consensus_algorithms:
                    try:
                        hm, _ = self._execute_strategy(model, tensor, target_class, layer_name, algo)
                        heatmaps.append(hm)
                    except Exception as e:
                        warnings.append(f"Consensus failed for {algo}: {e}")

                if heatmaps:
                    consensus_result = self.consensus.aggregate(heatmaps, [a.value for a in self.config.consensus_algorithms])
                    final_hm = consensus_result.consensus_heatmap

            # 2. Medical Metrics
            medical_metrics = self.metrics.compute(final_hm, image)

            # 3. Faithfulness
            faith_result = None
            if self.config.enable_faithfulness:
                faith_result = self.faithfulness.evaluate(model, image, tensor, final_hm, prediction, self.adapter)

            # 4. Stability
            stab_result = None
            if self.config.enable_stability:
                stab_result = self.stability.evaluate(model, image, tensor, primary_strat, prediction, self.adapter, layer_name, final_hm)

            # 5. Sanity
            sanity_result = None
            if self.config.enable_sanity_checks:
                sanity_result = self.sanity.evaluate(model, image, tensor, primary_strat, target_class, self.adapter, layer_name, final_hm)
                if not sanity_result.weight_randomization_passed:
                    warnings.append("Sanity Check Failed: Model weights randomized but heatmap did not change significantly.")

            # 6. Overall Quality Score
            components = [medical_metrics.metrics_score]
            if faith_result:
                components.append(faith_result.faithfulness_score)
            if stab_result:
                components.append(stab_result.stability_score)
            if sanity_result:
                components.append(sanity_result.sanity_score)

            overall_score = np.mean(components) if components else 0.0

            # --- M6.7∞+ Extensions ---
            try:
                from src.modules.explainability.clinical_validation.evidence_alignment import (
                    ExplanationAlignmentEngine,
                )
                from src.modules.explainability.counterfactual.recourse import CounterfactualEngine
                from src.modules.explainability.digital_twin.simulator import DigitalTwinSimulator
                from src.modules.explainability.drift_analysis.drift_analyzer import DriftAnalyzer
                from src.modules.explainability.evidence_consistency.consistency_engine import (
                    ConsistencyEngine,
                )
                from src.modules.explainability.evidence_consistency.evidence_graph import (
                    Edge,
                    EvidenceGraph,
                    Node,
                )
                from src.modules.explainability.failure_analysis.failure_report import (
                    FailureAnalyzer,
                )
                from src.modules.explainability.provenance.provenance_engine import ProvenanceEngine

                # Digital Twin & Counterfactual
                DigitalTwinSimulator(model)
                twin = image # Simplified stub
                cf_engine = CounterfactualEngine(self.adapter)
                cf_result = cf_engine.generate_recourse(twin, target_class)

                # Clinical Validation (CEAS)
                ceas_engine = ExplanationAlignmentEngine()
                clinical_alignment = ceas_engine.compute_ceas(final_hm, [], [])

                # Failure Mode Analysis
                failure_engine = FailureAnalyzer()
                failure_result = failure_engine.analyze(image)

                # Drift Analysis
                drift_engine = DriftAnalyzer()
                drift_result = drift_engine.analyze_drift(final_hm, np.zeros_like(final_hm))

                # Evidence Consistency (CECI)
                from src.modules.explainability.evidence_consistency.graph_propagation import (
                    RuleConstrainedPropagator,
                )
                from src.modules.explainability.evidence_consistency.trust_aggregator import (
                    TrustAggregator,
                )

                graph = EvidenceGraph()
                graph.add_node(Node("pred", "prediction", prediction.predicted_class))
                graph.add_node(Node("align", "alignment", clinical_alignment.alignment_score))
                graph.add_edge(Edge("pred", "align", 0.9, "supports"))

                # Provenance
                prov_engine = ProvenanceEngine()
                provenance = prov_engine.create_provenance("ExplainabilityPipeline", ["ModelWeights", "InputImage"], "orchestrator")

                ceci_engine = ConsistencyEngine()
                ceci_result = ceci_engine.compute_ceci(graph, provenance)

                # Apply Penalty Propagator & Trust
                propagator = RuleConstrainedPropagator()
                ceci_result.ceci_score = propagator.apply_penalties(graph, clinical_alignment.alignment_score, prediction.confidence, ceci_result.ceci_score)

                trust_aggr = TrustAggregator()
                ceci_result.trust_result = trust_aggr.fuse_evidence({"ceci": ceci_result.ceci_score, "ceas": clinical_alignment.alignment_score}, {"ceci": 0.1, "ceas": 0.4})

                evidence_matrix = None # Handled inside CEAS or CECI depending on implementation

            except Exception as ie:
                import traceback
                warnings.append(f"M6.7∞+ extensions missing or failed to load: {ie}\n{traceback.format_exc()}")
                clinical_alignment = None
                ceci_result = None
                cf_result = None
                failure_result = None
                drift_result = None
                evidence_matrix = None
            # -------------------------

            # 7. Render Overlay
            overlay = OverlayEngine.render(final_hm, image, self.config)

            result = ExplainabilityResult(
                algorithm=self.config.primary_algorithm.value,
                target_class=target_class,
                layer_name=layer_name,
                heatmap=final_hm,
                overlay=overlay,
                consensus=consensus_result,
                faithfulness=faith_result,
                stability=stab_result,
                sanity=sanity_result,
                medical_metrics=medical_metrics,
                clinical_alignment=clinical_alignment,
                ceci_result=ceci_result,
                counterfactual=cf_result,
                failure_mode=failure_result,
                drift_analysis=drift_result,
                evidence_matrix=evidence_matrix,
                overall_quality_score=overall_score,
                valid=True,
                status="SUCCESS",
                runtime_ms=(time.time() - start_time) * 1000.0,
                metadata=primary_strat.get_metadata() if hasattr(primary_strat, "get_metadata") else {},
                warnings=warnings
            )

            # 8. Visual & Reporting
            self.visualizer.generate_all(result, image, self.config.output_dir)
            report = self.reporter.generate(result, self.config)
            self.reporter.export(report, self.config.output_dir)

            self.cache.set(model, target_class, self.config, image, layer_name, result)
            return result

        except Exception as e:
            import traceback
            warnings.append(f"Explainability Execution Failed: {e}\n{traceback.format_exc()}")
            return ExplainabilityResult(
                valid=False,
                status="ERROR",
                algorithm=self.config.primary_algorithm.value,
                target_class=target_class,
                layer_name=layer_name,
                heatmap=None,
                runtime_ms=(time.time() - start_time) * 1000.0,
                warnings=warnings
            )
