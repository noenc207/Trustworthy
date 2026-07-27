
import torch
import torch.nn as nn


class FeatureExtractor:
    def __init__(self, model: nn.Module):
        self.model = model
        self.features = None
        self.hook = None

    def _hook_fn(self, module, input, output):
        self.features = output

    def extract_intermediate_layers(self, layer_name: str, x: torch.Tensor) -> torch.Tensor:
        for name, module in self.model.named_modules():
            if name == layer_name:
                self.hook = module.register_forward_hook(self._hook_fn)
                break

        if self.hook is None:
            raise ValueError(f"Layer {layer_name} not found")

        with torch.no_grad():
            self.model(x)

        self.hook.remove()
        return self.features

class EmbeddingExtractor:
    @staticmethod
    def extract_embedding(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Based on TrustworthyModel design which returns (logits, embeddings)
            if hasattr(model, 'forward'):
                res = model(x)
                if isinstance(res, tuple) and len(res) == 2:
                    return res[1]
                return res # Fallback
            return model(x)
