"""
Weight Resolution and Verification.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Protocol

class WeightResolverProtocol(Protocol):
    """Interface for resolving and validating model weights."""
    
    def resolve(self, model_id: str, version: str) -> Path:
        """Resolve model identifier to a physical path."""
        ...
        
    def validate_checksum(self, path: Path, expected_checksum: str) -> bool:
        """Validate integrity of the weight file."""
        ...

class LocalWeightResolver:
    """Implementation for local file system resolution."""
    
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def resolve(self, model_id: str, version: str) -> Path:
        path = self.base_dir / model_id / version / "weights.bin"
        if not path.exists():
            raise FileNotFoundError(f"Weights not found for {model_id} v{version} at {path}")
        return path

    def validate_checksum(self, path: Path, expected_checksum: str) -> bool:
        if not expected_checksum:
            return True
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest() == expected_checksum
