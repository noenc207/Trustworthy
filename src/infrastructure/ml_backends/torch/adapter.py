from pathlib import Path
from typing import Any
import torch

from src.modules.classifier.interfaces import BackendAdapter
from src.modules.classifier.exceptions import CorruptedWeightsError

class TorchBackendAdapter(BackendAdapter):
    """PyTorch Backend Adapter with Explainability Hook Support."""
    
    def load_model(self, path: Path, device: Any) -> torch.nn.Module:
        try:
            if not path.exists():
                raise FileNotFoundError(f"Model file not found at {path}")
            model = torch.jit.load(str(path))
            model.eval()
            return model
        except Exception as e:
            raise CorruptedWeightsError(f"Failed to load PyTorch model: {e}")
            
    def to_device(self, model: torch.nn.Module, device: Any) -> torch.nn.Module:
        if device:
            model = model.to(device)
        return model
        
    def predict(self, model: torch.nn.Module, inputs: Any) -> Any:
        import numpy as np
        if isinstance(inputs, np.ndarray):
            tensor = torch.from_numpy(inputs)
        else:
            tensor = inputs
            
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
            
        tensor = tensor.to(device)
        
        with torch.inference_mode():
            outputs = model(tensor)
            
        return outputs.cpu().numpy()

    # --- Explainability Hook Support ---
    
    def _get_layer(self, model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
        """Iteratively gets a layer from a PyTorch model by string name (e.g., 'features.7')."""
        parts = layer_name.split(".")
        current = model
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif part.isdigit() and isinstance(current, (torch.nn.Sequential, torch.nn.ModuleList)):
                current = current[int(part)]
            else:
                raise ValueError(f"Layer '{layer_name}' not found in model.")
        return current

    def register_forward_hook(self, model: Any, layer_name: str, callback: Any) -> Any:
        layer = self._get_layer(model, layer_name)
        return layer.register_forward_hook(callback)
        
    def register_backward_hook(self, model: Any, layer_name: str, callback: Any) -> Any:
        layer = self._get_layer(model, layer_name)
        return layer.register_full_backward_hook(callback)
        
    def remove_hook(self, handle: Any) -> None:
        handle.remove()
        
    def backward_pass(self, model: Any, inputs: Any, target_class: int) -> None:
        """Executes a forward and targeted backward pass to generate gradients."""
        import numpy as np
        if isinstance(inputs, np.ndarray):
            tensor = torch.from_numpy(inputs)
        else:
            tensor = inputs
            
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device('cpu')
            
        tensor = tensor.to(device).requires_grad_(True)
        
        # Zero gradients explicitly to avoid accumulating from previous passes
        model.zero_grad()
        
        outputs = model(tensor)
        if outputs.dim() == 2:
            score = outputs[0, target_class]
        else:
            score = outputs[target_class]
            
        score.backward(retain_graph=True)
        
    def get_module(self, model: Any, layer_name: str) -> Any:
        return self._get_layer(model, layer_name)
        
    def clone_model(self, model: Any) -> Any:
        import copy
        return copy.deepcopy(model)
        
    def randomize_weights(self, model: Any) -> None:
        import torch.nn as nn
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.xavier_uniform_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
