"""
Model registry and factory for classification backbones.
"""
from dataclasses import dataclass
import torch.nn as nn
from src.modules.classification.classifier import SkinLesionClassifier

@dataclass(frozen=True)
class BackboneSpec:
    name: str
    timm_name: str
    default_image_size: int
    supports_features: bool = True

SUPPORTED_BACKBONES: dict[str, BackboneSpec] = {
    'resnet50': BackboneSpec('resnet50', 'resnet50', 224),
    'resnet101': BackboneSpec('resnet101', 'resnet101', 224),
    'efficientnet_b0': BackboneSpec('efficientnet_b0', 'efficientnet_b0', 224),
    'efficientnet_b4': BackboneSpec('efficientnet_b4', 'efficientnet_b4', 380),
    'efficientnetv2_s': BackboneSpec('efficientnetv2_s', 'tf_efficientnetv2_s', 384),
    'densenet121': BackboneSpec('densenet121', 'densenet121', 224),
    'densenet201': BackboneSpec('densenet201', 'densenet201', 224),
    'convnext_tiny': BackboneSpec('convnext_tiny', 'convnext_tiny', 224),
    'convnext_small': BackboneSpec('convnext_small', 'convnext_small', 224),
    'vit_base': BackboneSpec('vit_base', 'vit_base_patch16_224', 224),
    'vit_small': BackboneSpec('vit_small', 'vit_small_patch16_224', 224),
    'swin_tiny': BackboneSpec('swin_tiny', 'swin_tiny_patch4_window7_224', 224),
    'swin_small': BackboneSpec('swin_small', 'swin_small_patch4_window7_224', 224),
    'mobilenetv3_large': BackboneSpec('mobilenetv3_large', 'mobilenetv3_large_100', 224),
    'regnet_y_400mf': BackboneSpec('regnet_y_400mf', 'regnet_y_400mf', 224),
}

class ModelFactory:
    @staticmethod
    def create(
        backbone: str = 'efficientnet_b4',
        num_classes: int = 7,
        pretrained: bool = True,
        drop_rate: float = 0.3,
        freeze_backbone: bool = False,
        freeze_bn: bool = False,
        gradient_checkpointing: bool = False,
    ) -> nn.Module:
        spec = ModelFactory.get_backbone_spec(backbone)
        model = SkinLesionClassifier(
            backbone=spec.timm_name,
            num_classes=num_classes,
            pretrained=pretrained,
            drop_rate=drop_rate
        )
        
        if freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False
                
        if freeze_bn:
            for module in model.backbone.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
                        
        if gradient_checkpointing:
            if hasattr(model.backbone, 'set_grad_checkpointing'):
                model.backbone.set_grad_checkpointing(True)
                
        return model

    @staticmethod
    def get_backbone_spec(name: str) -> BackboneSpec:
        if name not in SUPPORTED_BACKBONES:
            raise ValueError(f"Backbone {name} not supported. Available: {list(SUPPORTED_BACKBONES.keys())}")
        return SUPPORTED_BACKBONES[name]
    
    @staticmethod  
    def list_available() -> list[str]:
        return list(SUPPORTED_BACKBONES.keys())
