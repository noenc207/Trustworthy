from collections.abc import Callable
from typing import Any

from src.modules.classifier.interfaces import BackendAdapter
from src.modules.explainability.exceptions import HookRegistrationError


class HookManager:
    """Safely manages forward and backward hooks via the BackendAdapter."""

    def __init__(self, backend_adapter: BackendAdapter, model: Any):
        self.adapter = backend_adapter
        self.model = model
        self._hook_handles = []
        self.activations = None
        self.gradients = None

    def _forward_hook_wrapper(self) -> Callable:
        def hook(module, input, output):
            # The backend adapter must ensure output is converted/accessible
            # For gradients, we need to ensure it retains grad.
            self.activations = output
        return hook

    def _backward_hook_wrapper(self) -> Callable:
        def hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        return hook

    def register_hooks(self, layer_name: str) -> None:
        """Registers both forward and backward hooks on the target layer."""
        if not hasattr(self.adapter, "register_forward_hook"):
            raise HookRegistrationError("BackendAdapter does not support register_forward_hook.")

        try:
            h_fw = self.adapter.register_forward_hook(self.model, layer_name, self._forward_hook_wrapper())
            h_bw = self.adapter.register_backward_hook(self.model, layer_name, self._backward_hook_wrapper())
            self._hook_handles.extend([h_fw, h_bw])
        except Exception as e:
            self.cleanup()
            raise HookRegistrationError(f"Failed to register hooks: {e}")

    def remove_hooks(self) -> None:
        """Explicitly remove all hooks from the model."""
        if hasattr(self.adapter, "remove_hook"):
            for handle in self._hook_handles:
                self.adapter.remove_hook(handle)
        self._hook_handles.clear()

    def cleanup(self) -> None:
        """Aggressively clear references to prevent memory leaks."""
        self.remove_hooks()
        self.activations = None
        self.gradients = None
        self.model = None
        self.adapter = None

    def get_activations(self) -> Any:
        return self.activations

    def get_gradients(self) -> Any:
        return self.gradients
