"""
Configuration Manager.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class FrameworkConfig:
    """Immutable runtime framework configuration."""
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"
    model_registry_path: str = "models/"
    device: str = "cpu"

class ConfigurationManager:
    """
    Configuration manager with YAML loading and environment overrides.
    Designed for future Hydra compatibility.
    """

    @staticmethod
    def load(path: Path | str) -> FrameworkConfig:
        path_obj = Path(path)
        data = {}
        if path_obj.exists():
            with open(path_obj, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        # Support basic environment overrides
        env = os.getenv("APP_ENV", data.get("environment", "development"))
        debug = os.getenv("APP_DEBUG", str(data.get("debug", False))).lower() == "true"
        log_level = os.getenv("APP_LOG_LEVEL", data.get("log_level", "INFO"))
        device = os.getenv("APP_DEVICE", data.get("device", "cpu"))

        return FrameworkConfig(
            environment=env,
            debug=debug,
            log_level=log_level,
            device=device,
            model_registry_path=data.get("model_registry_path", "models/")
        )
