"""
Evaluate OOD Robustness using Energy Score.
"""
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import argparse
import numpy as np
from pathlib import Path
from loguru import logger
import torch
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load
from omegaconf import DictConfig, ListConfig, OmegaConf

try:
    from torch.serialization import add_safe_globals
    import omegaconf.base; import omegaconf.nodes; import omegaconf.base; import omegaconf.nodes; import typing; add_safe_globals([DictConfig, ListConfig, omegaconf.base.ContainerMetadata, omegaconf.nodes.AnyNode, typing.Any])
except ImportError:
    pass

from src.training.train_pipeline import SkinLesionLightningModule
from src.training.data_module import SkinLesionDataModule

def compute_energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Compute Energy Score: -T * log(sum(exp(logits / T)))"""
    return -T * torch.logsumexp(logits / T, dim=-1)

def evaluate_ood(checkpoint_path: str, ood_dataset_path: str = None):
    logger.info("Evaluating OOD Robustness (Energy-based)...")
    
    # 1. Load ID logits from test split
    results_dir = Path("research/baseline_v2/results")
    id_preds_file = results_dir / "predictions_baseline_v2_test.npz"
    if not id_preds_file.exists():
        logger.error(f"Cannot find ID predictions at {id_preds_file}. Run evaluate script first.")
        raise FileNotFoundError(f"Missing {id_preds_file}")
        
    id_data = np.load(id_preds_file)
    id_logits = torch.tensor(id_data["logits"], dtype=torch.float32)
    id_energy = compute_energy_score(id_logits).numpy()
    
    logger.info(f"Computed Energy Scores for {len(id_energy)} ID samples.")
    
    # 2. Check if OOD dataset exists
    if not ood_dataset_path or not Path(ood_dataset_path).exists():
        logger.warning("No real OOD dataset provided or found.")
        logger.info("OOD evaluation unavailable")
        # We save just the ID scores
        np.savez(
            results_dir / "ood_scores.npz",
            id_energy=id_energy
        )
        return
        
    # If an OOD dataset exists, we would load it and run inference
    logger.info(f"OOD Dataset found at {ood_dataset_path}. Running inference...")
    # Real OOD inference logic would go here
    # Since we are in smoke test and no OOD dataset is provided yet, this branch is technically unreachable without args
    logger.info("OOD evaluation unavailable (mocked out until real dataset config)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--ood_dataset", type=str, default=None)
    args = parser.parse_args()
    evaluate_ood(args.checkpoint, args.ood_dataset)



