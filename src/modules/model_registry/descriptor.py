"""
Model metadata descriptor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModelDescriptor:
    """Immutable record containing all metadata for a registered model."""
    model_id: str
    version: str
    author: str
    backend: str
    task: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    input_specification: dict[str, str] = field(default_factory=dict)
    output_specification: dict[str, str] = field(default_factory=dict)
    supported_devices: list[str] = field(default_factory=list)
    preprocessing_requirements: dict[str, str] = field(default_factory=dict)
    compatibility_information: dict[str, str] = field(default_factory=dict)
    checksum: str | None = None
