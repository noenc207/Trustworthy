from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaithfulnessResult:
    insertion_auc: float = 0.0
    deletion_auc: float = 0.0
    average_drop: float = 0.0
    average_increase: float = 0.0
    prediction_preservation: float = 0.0
    confidence_change: float = 0.0
    faithfulness_score: StatisticalResult | None = None # 0-100

@dataclass
class StabilityResult:
    ssim: float = 0.0
    iou: float = 0.0
    dice: float = 0.0
    correlation: float = 0.0
    mad: float = 0.0
    mse: float = 0.0
    stability_score: StatisticalResult | None = None # 0-100

@dataclass
class SanityResult:
    weight_randomization_passed: bool = False
    label_randomization_passed: bool = False
    heatmap_ssim_drop: float = 0.0
    sanity_score: StatisticalResult | None = None # 0-100

@dataclass
class MedicalMetrics:
    lesion_coverage: float = 0.0
    boundary_coverage: float = 0.0
    edge_alignment: float = 0.0
    compactness: float = 0.0
    connected_components: float = 0.0
    sparsity: float = 0.0
    density: float = 0.0
    center_of_mass: tuple[float, float] | None = None
    peak_activation: float = 0.0
    entropy: float = 0.0
    attention_dispersion: float = 0.0
    focus_score: float = 0.0
    localization_score: float = 0.0
    noise_score: float = 0.0
    clinical_relevance_score: float = 0.0
    background_leakage: float = 0.0
    false_attention_ratio: float = 0.0
    metrics_score: StatisticalResult | None = None # 0-100

@dataclass
class ConsensusResult:
    agreement_score: float = 0.0 # 0-100
    disagreement_map: np.ndarray | None = None
    consensus_heatmap: np.ndarray | None = None
    algorithms_used: list[str] = field(default_factory=list)

@dataclass
class EvidenceProvenance:
    origin: str
    dependencies: list[str]
    version: str
    checksum: str
    timestamp: str
    source_module: str

@dataclass
class StatisticalResult:
    point_estimate: float
    ci_95: tuple[float, float]
    standard_error: float
    bootstrap_samples: int
    confidence_level: float
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    ci_width: float = 0.0
    coefficient_of_variation: float = 0.0
    bootstrap_method: str = "BCa"

@dataclass
class OntologyRecord:
    who_classification: str
    icd_code: str
    dermnet_taxonomy: str
    related_concepts: list[str]
    is_unknown: bool = False

@dataclass
class RuleResult:
    rule_name: str
    status: str
    clinical_notes: str

@dataclass
class ClinicalAlignmentResult:
    alignment_score: float
    alignment_ci: tuple[float, float]
    rule_matches: list[RuleResult]
    rule_violations: list[RuleResult]
    literature_alignment: list[OntologyRecord]
    morphology_alignment: dict[str, float]

@dataclass
class TrustResult:
    trust_score: float
    trust_variance: float
    trust_ci: tuple[float, float]
    evidence_contribution: dict[str, float]

@dataclass
class CECIResult:
    graph_consistency: float
    ceci_score: float
    ceci_stats: StatisticalResult
    provenance: EvidenceProvenance
    recommendation: str
    trust_result: TrustResult | None = None

@dataclass
class CounterfactualRecourse:
    modifications: list[str]
    new_probabilities: dict[str, float]
    estimated_distance: float

@dataclass
class FailureModeResult:
    failure_probability: float
    primary_cause: str
    secondary_cause: str
    artifact_confidence: float
    recommended_action: str

@dataclass
class DriftResult:
    status: str # Stable, Minor Drift, Major Drift
    kl_divergence: float
    js_divergence: float
    earth_mover_distance: float
    ssim: float
    cosine_similarity: float
    pearson_correlation: float

@dataclass
class ExplainabilityResult:
    """Consolidated result from the Explainability Pipeline."""
    valid: bool
    status: str
    algorithm: str
    target_class: int
    layer_name: str
    heatmap: np.ndarray | None
    normalized_heatmap: np.ndarray | None = None
    overlay: np.ndarray | None = None
    runtime_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # Evidence Layers
    consensus: ConsensusResult | None = None
    faithfulness: FaithfulnessResult | None = None
    stability: StabilityResult | None = None
    sanity: SanityResult | None = None
    medical_metrics: MedicalMetrics | None = None

    # M6.7∞+ Extensions
    clinical_alignment: ClinicalAlignmentResult | None = None
    ceci_result: CECIResult | None = None
    counterfactual: CounterfactualRecourse | None = None
    evidence_matrix: np.ndarray | None = None

    # Final Scientific Revision
    failure_mode: FailureModeResult | None = None
    drift_analysis: DriftResult | None = None

    warnings: list[str] = field(default_factory=list)
    overall_quality_score: float = 0.0 # 0-100

@dataclass
class ClinicalReport:
    executive_summary: str
    clinical_interpretation: str
    overall_quality_score: float
    is_reliable: bool
    metrics_breakdown: dict[str, float]
    warnings: list[str]
    recommendations: list[str]
