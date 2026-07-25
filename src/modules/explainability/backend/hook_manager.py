import contextlib
import traceback
from typing import Any

from src.infrastructure.ml_backends.torch.adapter import TorchBackendAdapter
from src.modules.explainability.exceptions import HookRegistrationError
from src.modules.explainability.interfaces import HookManagerInterface


class HookManager(HookManagerInterface):
    """Orchestrates secure tensor capture from backends without exposing PyTorch globally."""

    def __init__(self, adapter: TorchBackendAdapter, model: Any):
        self.adapter = adapter
        self.model = model
        self.activations: Any | None = None
        self.gradients: Any | None = None
        self._hook_handles = []

    def register_hooks(self, layer_name: str) -> None:
        target_layer = self.adapter.get_module(self.model, layer_name)
        if target_layer is None:
            raise HookRegistrationError(f"Target layer '{layer_name}' not found in model.")

        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        try:
            fh = self.adapter.register_forward_hook(self.model, layer_name, forward_hook)
            bh = self.adapter.register_backward_hook(self.model, layer_name, backward_hook)
            self._hook_handles.extend([fh, bh])
        except Exception as e:
            self.cleanup()
            raise HookRegistrationError(f"Failed to register hooks on '{layer_name}': {e}\n{traceback.format_exc()}")

    def get_activations(self) -> Any:
        return self.activations

    def get_gradients(self) -> Any:
        return self.gradients

    def cleanup(self) -> None:
        for handle in self._hook_handles:
            with contextlib.suppress(Exception):
                handle.remove()
        self._hook_handles.clear()
        self.activations = None
        self.gradients = None
