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
        
    # If an OOD dataset exists, load and run real inference
    logger.info(f"OOD Dataset found at {ood_dataset_path}. Running inference...")
    
    # Load OOD dataset using the same transforms as test data
    from src.training.augmentation import get_val_transforms
    from src.modules.dataset_manager.pytorch_dataset import SkinLesionDataset
    from torch.utils.data import DataLoader
    
    ood_path = Path(ood_dataset_path)
    ood_csv = ood_path / "labels" / "cleaned.csv"
    if not ood_csv.exists():
        raise FileNotFoundError(f"OOD dataset CSV not found at {ood_csv}")
    
    ood_dataset = SkinLesionDataset(cleaned_csv_path=ood_csv, transform=get_val_transforms(224), image_size=224)
    ood_loader = DataLoader(ood_dataset, batch_size=32, shuffle=False)
    
    # Load model from checkpoint for OOD inference
    model_module = SkinLesionLightningModule.load_from_checkpoint(checkpoint_path, strict=True)
    model = model_module.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    ood_logits_list = []
    with torch.no_grad():
        for batch in ood_loader:
            x, y = batch[0], batch[1]
            x = x.to(device)
            logits = model(x)
            ood_logits_list.append(logits.cpu())
    
    ood_logits = torch.cat(ood_logits_list, dim=0)
    ood_energy = compute_energy_score(ood_logits).numpy()
    
    logger.info(f"Computed Energy Scores for {len(ood_energy)} OOD samples.")
    
    # Compute OOD AUROC
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.ones(len(id_energy)), np.zeros(len(ood_energy))])
    scores = np.concatenate([-id_energy, -ood_energy])  # Higher energy = more OOD, negate for AUROC convention
    ood_auroc = roc_auc_score(labels, scores)
    
    logger.info(f"OOD AUROC (Energy): {ood_auroc:.4f}")
    
    np.savez(
        results_dir / "ood_scores.npz",
        id_energy=id_energy,
        ood_energy=ood_energy,
        ood_auroc=ood_auroc
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--ood_dataset", type=str, default=None)
    args = parser.parse_args()
    evaluate_ood(args.checkpoint, args.ood_dataset)



