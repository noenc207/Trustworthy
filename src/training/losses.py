"""Loss functions for the training engine."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Any

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(inputs, targets, weight=self.weight, label_smoothing=self.smoothing, reduction=self.reduction)

class WeightedCrossEntropy(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: str = 'mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(inputs, targets, weight=self.weight, reduction=self.reduction)

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        
        intersection = (inputs * targets_one_hot).sum(dim=0)
        cardinality = inputs.sum(dim=0) + targets_one_hot.sum(dim=0)
        
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1 - dice.mean()

class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).float()
        
        tp = (inputs * targets_one_hot).sum(dim=0)
        fp = (inputs * (1 - targets_one_hot)).sum(dim=0)
        fn = ((1 - inputs) * targets_one_hot).sum(dim=0)
        
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 1.0, clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        y_one_hot = F.one_hot(y, num_classes=x.shape[1]).float()
        x = torch.sigmoid(x)
        
        xs_pos = x
        xs_neg = 1 - x
        
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
            
        los_pos = y_one_hot * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y_one_hot) * torch.log(xs_neg.clamp(min=self.eps))
        
        loss_pos = los_pos * (1 - xs_pos) ** self.gamma_pos
        loss_neg = los_neg * (1 - xs_neg) ** self.gamma_neg
        
        return -(loss_pos + loss_neg).sum(dim=1).mean()

class CompositeLoss(nn.Module):
    def __init__(self, losses: List[nn.Module], weights: List[float]):
        super().__init__()
        self.losses = nn.ModuleList(losses)
        self.weights = weights

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=inputs.device)
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        return total_loss

class LossFactory:
    @staticmethod
    def create(name: str, **kwargs: Any) -> nn.Module:
        name = name.lower()
        if name == 'focal':
            return FocalLoss(**kwargs)
        elif name == 'label_smoothing':
            return LabelSmoothingCrossEntropy(**kwargs)
        elif name == 'cross_entropy':
            return WeightedCrossEntropy(**kwargs)
        elif name == 'dice':
            return DiceLoss(**kwargs)
        elif name == 'tversky':
            return TverskyLoss(**kwargs)
        elif name == 'asymmetric':
            return AsymmetricLoss(**kwargs)
        elif name == 'composite':
            losses = kwargs.pop('losses', [])
            weights = kwargs.pop('weights', [])
            return CompositeLoss(losses, weights)
        else:
            raise ValueError(f"Unknown loss function: {name}")
