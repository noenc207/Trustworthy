import torch
import torch.nn as nn
from .builder import BackboneBuilder, ClassifierBuilder

class TrustworthyModel(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head
        
    def forward(self, x):
        emb = self.backbone(x)
        if emb.ndim > 2:
            emb = emb.view(emb.size(0), -1)
        logits = self.head(emb)
        return logits, emb

class ModelFactory:
    @staticmethod
    def create(backbone_name: str, num_classes: int, pretrained: bool = True, classifier_name: str = "linear", dropout: float = 0.0) -> TrustworthyModel:
        # Build backbone
        backbone = BackboneBuilder.build(backbone_name, pretrained)
        in_features = getattr(backbone, "out_features", 1000)
        
        # Build head
        head = ClassifierBuilder.build(classifier_name, in_features, num_classes, dropout)
        
        return TrustworthyModel(backbone, head)
