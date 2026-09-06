from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class CounterfactualEngine:
    """Generates counterfactual inputs and measures prediction sensitivity.
    
    For each evidence region k:
        x^{-k} = mask/replace region k in image
        Δp_k = p(y|x) - p(y|x^{-k})
        S_k = D_KL(p(y|x) || p(y|x^{-k}))
    """
    def __init__(self, model: nn.Module, num_regions: int = 8, image_size: int = 224):
        self.model = model
        self.num_regions = num_regions
        self.image_size = image_size
    
    def generate_counterfactual(self, image_tensor: torch.Tensor, region_idx: int) -> torch.Tensor:
        """Create counterfactual by masking region k with mean pixel value."""
        cf_image = image_tensor.clone()
        B, C, H, W = cf_image.shape
        
        # Simple grid masking logic for 8 regions (2x3 grid = 6, + center = 1, + border = 1)
        if region_idx < 6:
            row = region_idx // 3
            col = region_idx % 3
            h_step = H // 2
            w_step = W // 3
            cf_image[:, :, row*h_step:(row+1)*h_step, col*w_step:(col+1)*w_step] = 0.0
        elif region_idx == 6:
            # Center circle approximation
            center_h, center_w = H // 2, W // 2
            radius = min(H, W) // 4
            Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            dist = (Y - center_h)**2 + (X - center_w)**2
            mask = dist < radius**2
            cf_image[:, :, mask] = 0.0
        elif region_idx == 7:
            # Border ring approximation
            center_h, center_w = H // 2, W // 2
            radius_inner = min(H, W) * 3 // 8
            radius_outer = min(H, W) // 2
            Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            dist = (Y - center_h)**2 + (X - center_w)**2
            mask = (dist > radius_inner**2) & (dist < radius_outer**2)
            cf_image[:, :, mask] = 0.0
            
        return cf_image
    
    @torch.no_grad()
    def compute_sensitivity(self, image_tensor: torch.Tensor) -> dict[int, dict[str, float]]:
        """Compute prediction sensitivity to each region.
        Returns dict mapping region_idx -> {delta_p, kl_div, confidence_change}."""
        base_logits = self.model(image_tensor)
        base_probs = torch.softmax(base_logits, dim=-1)
        base_conf, base_pred = torch.max(base_probs, dim=-1)
        
        sensitivities = {}
        for k in range(self.num_regions):
            cf_image = self.generate_counterfactual(image_tensor, k)
            cf_logits = self.model(cf_image)
            cf_probs = torch.softmax(cf_logits, dim=-1)
            cf_conf, _ = torch.max(cf_probs, dim=-1)
            
            delta_p = (base_probs[0, base_pred[0]] - cf_probs[0, base_pred[0]]).item()
            
            kl_div = F.kl_div(cf_probs.log(), base_probs, reduction='batchmean').item()
            
            conf_change = (base_conf[0] - cf_conf[0]).item()
            
            sensitivities[k] = {
                "delta_p": delta_p,
                "kl_div": kl_div,
                "confidence_change": conf_change
            }
            
        return sensitivities
