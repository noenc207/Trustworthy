"""Learning rate scheduler framework for the training engine."""

from typing import Any

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LRScheduler


class WarmupScheduler(LRScheduler):
    def __init__(self, optimizer: torch.optim.Optimizer, base_scheduler: LRScheduler, warmup_epochs: int, multiplier: float = 1.0):
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.multiplier = multiplier
        super().__init__(optimizer, last_epoch=-1)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * self.multiplier * alpha for base_lr in self.base_lrs]

        self.base_scheduler.last_epoch = self.last_epoch - self.warmup_epochs
        return self.base_scheduler.get_lr()

    def step(self, epoch: Any = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.last_epoch < self.warmup_epochs:
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        else:
            self.base_scheduler.step(epoch - self.warmup_epochs)

class CosineWarmRestarts(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer: torch.optim.Optimizer, T_0: int, T_mult: int = 1, eta_min: float = 0, last_epoch: int = -1, warmup_epochs: int = 0):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        return super().get_lr()

class SchedulerFactory:
    @staticmethod
    def create(name: str, optimizer: torch.optim.Optimizer, **kwargs: Any) -> Any:
        name = name.lower()
        if name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        elif name == 'cosine_warm_restarts':
            return CosineWarmRestarts(optimizer, **kwargs)
        elif name == 'onecycle':
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, **kwargs)
        elif name == 'step':
            return torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif name == 'multistep':
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
        elif name == 'plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
        elif name == 'polynomial':
            return torch.optim.lr_scheduler.PolynomialLR(optimizer, **kwargs)
        else:
            raise ValueError(f"Unknown scheduler: {name}")
