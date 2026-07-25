"""Exponential Moving Average (EMA) for models."""

import torch
import torch.nn as nn
from typing import Dict, Any

class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.original_params: Dict[str, torch.Tensor] = {}
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

    def update(self) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow_params[name]
                    self.shadow_params[name].copy_(new_average)

    def apply(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.original_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.original_params:
                    param.data.copy_(self.original_params[name])
        self.original_params = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow_params

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        for name, param in state_dict.items():
            if name in self.shadow_params:
                self.shadow_params[name].copy_(param)
