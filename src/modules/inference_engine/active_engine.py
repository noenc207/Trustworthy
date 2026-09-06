"""
Active Inference Engine for DERMA-ACT.

Implements the active perception state machine on top of the Blackboard pattern.
Replaces the linear pipeline with a cyclic evidence acquisition loop.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from src.modules.active_perception.action_space import ObservationAction, get_valid_actions
from src.modules.active_perception.evidence_state import EvidenceState
from src.modules.active_perception.next_best_observation import NextBestObservationPolicy
from src.modules.active_perception.observation_encoder import ObservationEncoder
from src.modules.active_perception.view_generator import ViewGenerator
from src.modules.classification.classifier import SkinLesionClassifier
from src.modules.dynamic_routing.expert_router import DynamicExpertRouter
from src.modules.dynamic_routing.fusion import ExpertFusion
from src.modules.inference_engine.context import PipelineContext
from src.modules.selective_decision.risk_controller import RiskController, SelectiveDecision
from src.modules.selective_decision.stopping_policy import StoppingPolicy
from src.modules.self_critique.fragility import DecisionFragility

logger = logging.getLogger(__name__)


class ActiveInferenceEngine:
    """
    DERMA-ACT active evidence acquisition orchestrator.
    
    State Machine:
    INITIALIZE -> OBSERVE -> CRITIQUE -> SELECT_ACTION -> (loop if CONTINUE) -> RISK_CHECK -> DECIDE
    """

    def __init__(
        self,
        classifier,
        view_generator: ViewGenerator,
        observation_encoder: ObservationEncoder,
        router: DynamicExpertRouter | None,
        fusion: ExpertFusion | None,
        policy: NextBestObservationPolicy,
        stopping_policy: StoppingPolicy,
        risk_controller: RiskController,
        fragility_engine: DecisionFragility | None,
        device: str = "cpu",
    ):
        self.classifier = classifier
        self.view_generator = view_generator
        self.observation_encoder = observation_encoder
        self.router = router
        self.fusion = fusion
        self.policy = policy
        self.stopping_policy = stopping_policy
        self.risk_controller = risk_controller
        self.fragility_engine = fragility_engine
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def run(self, context: PipelineContext) -> PipelineContext:
        logger.info(f"Starting active inference loop (budget: {context.evidence_budget})")
        image_array = context.raw_image
        context.observation_history = []
        context.active_state = None
        current_action = ObservationAction.KEEP_FULL
        step = 0
        
        while step <= context.evidence_budget:
            try:
                view = self.view_generator.generate(image_array, current_action)
            except Exception as e:
                context.add_error(f"Failed to generate view for {current_action}: {str(e)}")
                break
                
            view_tensor = self._preprocess(view)
            
            with torch.no_grad():
                if isinstance(self.classifier, (list, tuple)):
                    features = [model.extract_features(view_tensor) for model in self.classifier]
                    logits = [model.head(f) for model, f in zip(self.classifier, features)]
                    if self.router and self.fusion:
                        weights, _ = self.router(features)
                        fused_logits = self.fusion(logits, weights)
                        probs = torch.softmax(fused_logits, dim=-1)
                        context.classification = probs
                        combined_features = torch.cat(features, dim=1)
                    else:
                        probs = torch.softmax(torch.mean(torch.stack(logits), dim=0), dim=-1)
                        combined_features = features[0]
                else:
                    # Generic single model forward
                    features = self.classifier(view_tensor) # Dummy logic assuming classifier returns features
                    probs = torch.softmax(features, dim=-1)
                    combined_features = features
            
            action_idx = current_action.value
            context.observation_history.append((action_idx, combined_features))
            
            seq_features = [f for _, f in context.observation_history]
            seq_actions = [a for a, _ in context.observation_history]
            h_t = self.observation_encoder.encode_sequence(seq_features, seq_actions)
            
            state = EvidenceState(
                h_t=h_t.squeeze(0),
                p_t=probs.squeeze(0),
                u_epistemic=0.0,
                u_aleatoric=0.0,
                ood_score=0.0,
                quality_score=1.0,
                evidence_support=torch.zeros(8, device=self.device),
                evidence_contradiction=torch.zeros(8, device=self.device),
                evidence_missing=torch.zeros(8, device=self.device),
                observation_history=seq_actions,
                budget_remaining=context.evidence_budget - step,
                step=step
            )
            context.active_state = state
            
            state_tensor = state.to_tensor().unsqueeze(0)
            with torch.no_grad():
                v_stop, v_cont = self.stopping_policy(state_tensor)
                
            if step >= context.evidence_budget:
                break
                
            if self.stopping_policy.should_stop(state_tensor):
                break
                
            valid_actions = get_valid_actions([ObservationAction(a) for a in seq_actions], state.budget_remaining)
            valid_mask = torch.zeros(13, device=self.device)
            for a in valid_actions:
                valid_mask[a.value] = 1.0
                
            with torch.no_grad():
                next_action_idx = self.policy.select_action(state_tensor, valid_mask.unsqueeze(0))
                current_action = ObservationAction(next_action_idx)
                
            if current_action == ObservationAction.STOP:
                break
            elif current_action == ObservationAction.ABSTAIN:
                context.selective_decision = SelectiveDecision.ABSTAIN.value
                break
                
            step += 1

        if context.selective_decision != SelectiveDecision.ABSTAIN.value:
            context.selective_decision = self.risk_controller.decide(
                ood_score=0.0,
                fragility=0.0,
                conformal_set={int(torch.argmax(context.active_state.p_t).item())},
                uncertainty=0.0
            ).value
            
        return context

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        img_float = image.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return ((tensor - mean) / std).to(self.device)
