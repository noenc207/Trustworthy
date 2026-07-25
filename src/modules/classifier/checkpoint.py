import torch
from pathlib import Path
from typing import Dict, Any, Optional
import hashlib

from .exceptions import CheckpointError
from .dto import CheckpointMetadata

class CheckpointManager:
    @staticmethod
    def _compute_sha256(filepath: str) -> str:
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def save_checkpoint(model: torch.nn.Module, filepath: str, metadata: CheckpointMetadata) -> None:
        try:
            state = {
                "model_state_dict": model.state_dict(),
                "metadata": metadata.__dict__
            }
            torch.save(state, filepath)
            
            # Verify save was successful
            if not Path(filepath).exists():
                raise CheckpointError(f"Failed to write checkpoint to {filepath}", "WRITE_FAILED")
        except Exception as e:
            raise CheckpointError(f"Serialization failed: {str(e)}", "SERIALIZATION_ERROR")

    @staticmethod
    def load_checkpoint(filepath: str, model: Optional[torch.nn.Module] = None, strict: bool = True, device: str = "cpu") -> Dict[str, Any]:
        if not Path(filepath).exists():
            raise CheckpointError(f"Checkpoint not found: {filepath}", "FILE_NOT_FOUND")
            
        try:
            state = torch.load(filepath, map_location=device, weights_only=False)
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {str(e)}", "LOAD_FAILED")
            
        if "model_state_dict" not in state:
            raise CheckpointError("Invalid checkpoint format: missing model_state_dict", "INVALID_FORMAT")
            
        if model is not None:
            try:
                model.load_state_dict(state["model_state_dict"], strict=strict)
            except Exception as e:
                raise CheckpointError(f"Architecture mismatch: {str(e)}", "ARCHITECTURE_MISMATCH")
                
        return state

    @staticmethod
    def load_weights_only(filepath: str, model: torch.nn.Module, strict: bool = True, device: str = "cpu") -> None:
        CheckpointManager.load_checkpoint(filepath, model, strict, device)
