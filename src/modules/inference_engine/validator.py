"""
Runtime consistency validator.
"""
from __future__ import annotations

from collections.abc import Sequence

from src.modules.inference_engine.context import PipelineConfig
from src.modules.inference_engine.stages import PipelineStage


class RuntimeValidator:
    """Validates pipeline and configuration integrity."""

    @staticmethod
    def validate_stages(stages: Sequence[PipelineStage]) -> None:
        """Check for duplicate stages."""
        seen = set()
        for s in stages:
            if s.name in seen:
                raise ValueError(f"Duplicate stage name found: {s.name}")
            seen.add(s.name)

    @staticmethod
    def validate_config(config: PipelineConfig | None) -> None:
        """Ensure configuration is present and valid."""
        if config is None:
            raise ValueError("Configuration snapshot is missing.")
        if not config.device_str:
            raise ValueError("Device string is missing from configuration.")
