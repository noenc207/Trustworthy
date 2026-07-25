"""
Backend-agnostic device manager.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import AppSettings


class DeviceManager:
    """Immutable runtime device information."""

    def __init__(self, device_str: str) -> None:
        self._device_str = device_str
        self._is_cpu = device_str.lower() == "cpu"
        self._is_cuda = device_str.lower().startswith("cuda")
        self._is_mps = device_str.lower() == "mps"

    @property
    def device_str(self) -> str:
        return self._device_str

    @property
    def is_cuda(self) -> bool:
        return self._is_cuda

    @property
    def is_cpu(self) -> bool:
        return self._is_cpu

    @property
    def is_mps(self) -> bool:
        return self._is_mps

    @classmethod
    def auto_detect(cls, preferred: str = "cuda") -> DeviceManager:
        """Auto detect device dynamically without top-level torch import."""
        if preferred.startswith("cuda"):
            try:
                import torch
                if torch.cuda.is_available():
                    return cls(preferred)
            except ImportError:
                import logging
                logging.getLogger(__name__).debug("torch not installed; cannot check cuda")
        elif preferred == "mps":
            try:
                import torch
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return cls("mps")
            except ImportError:
                import logging
                logging.getLogger(__name__).debug("torch not installed; cannot check mps")
        return cls("cpu")

    @classmethod
    def from_config(cls, settings: AppSettings) -> DeviceManager:
        """Create from app configuration."""
        return cls.auto_detect(settings.model.device)

    @classmethod
    def cpu(cls) -> DeviceManager:
        """Force CPU device."""
        return cls("cpu")

    def __repr__(self) -> str:
        return f"DeviceManager(device='{self._device_str}')"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DeviceManager):
            return False
        return self._device_str == other._device_str
