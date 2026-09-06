"""
DERMA-ACT Experiment Runner.
Executes the active perception loop using Hydra configuration.
"""
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import hydra
from omegaconf import DictConfig
from loguru import logger
import torch

from src.modules.active_perception.view_generator import ViewGenerator
from src.modules.active_perception.observation_encoder import ObservationEncoder
from src.modules.active_perception.next_best_observation import NextBestObservationPolicy
from src.modules.selective_decision.stopping_policy import StoppingPolicy
from src.modules.selective_decision.risk_controller import RiskController
from src.modules.classification.classifier import SkinLesionClassifier
from src.modules.inference_engine.active_engine import ActiveInferenceEngine
from src.modules.inference_engine.context import PipelineContext, PipelineConfig

@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    logger.info("Initializing DERMA-ACT framework...")
    
    # Initialize components
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    classifier = SkinLesionClassifier(cfg.model.backbone, num_classes=cfg.model.num_classes)
    classifier.to(device)
    classifier.eval()
    
    view_generator = ViewGenerator(image_size=224)
    obs_encoder = ObservationEncoder(feature_dim=cfg.model.get('feature_dim', 2048), hidden_dim=256).to(device)
    policy = NextBestObservationPolicy(state_dim=EvidenceState.state_dim(256), num_actions=13).to(device)
    stopping_policy = StoppingPolicy(state_dim=EvidenceState.state_dim(256)).to(device)
    risk_controller = RiskController(ood_threshold=0.5, fragility_threshold=0.8)
    
    engine = ActiveInferenceEngine(
        classifier=classifier,
        view_generator=view_generator,
        observation_encoder=obs_encoder,
        router=None,
        fusion=None,
        policy=policy,
        stopping_policy=stopping_policy,
        risk_controller=risk_controller,
        fragility_engine=None,
        device=device
    )
    
    logger.info("Ready for experiments.")
    return 0

if __name__ == "__main__":
    from src.modules.active_perception.evidence_state import EvidenceState
    main()
