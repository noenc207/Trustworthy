"""
Transform Registry.
Maps string identifiers to Albumentations transform classes.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from loguru import logger

# Global dictionary to hold registered transforms
TRANSFORMS_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_transform(name: str) -> Callable:
    """
    Decorator to register an Albumentations transform class or factory function
    into the global transforms registry.

    Args:
        name: The string identifier used in Hydra configs (e.g., 'dull_razor').
    """
    def wrapper(cls: Callable[..., Any]) -> Callable[..., Any]:
        if name in TRANSFORMS_REGISTRY:
            logger.warning(f"Transform '{name}' is already registered. Overwriting.")
        TRANSFORMS_REGISTRY[name] = cls
        return cls

    return wrapper


def get_transform_class(name: str) -> Callable[..., Any]:
    """Retrieve a transform class by its registered name."""
    if name not in TRANSFORMS_REGISTRY:
        raise KeyError(
            f"Transform '{name}' not found in registry. "
            f"Available transforms: {list(TRANSFORMS_REGISTRY.keys())}"
        )
    return TRANSFORMS_REGISTRY[name]
