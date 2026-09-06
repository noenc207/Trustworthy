"""
Evaluate Uncertainty using MC Dropout.
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
from src.modules.uncertainty.mc_dropout import enable_mc_dropout, compute_mutual_information

def evaluate_uncertainty(checkpoint_path: str, split: str = "test", n_passes: int = 30):
    logger.info(f"Evaluating Uncertainty (MC Dropout) with {n_passes} passes on {split} split...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model_module = SkinLesionLightningModule.load_from_checkpoint(checkpoint_path, strict=True)
        model = model_module.model
        cfg = model_module.hparams.cfg if hasattr(model_module.hparams, "cfg") else model_module.hparams
    except Exception as e:
        logger.error(f"Failed to load checkpoint strictly: {e}")
        raise RuntimeError(f"Checkpoint load failed: {e}")
        
    dm = SkinLesionDataModule(cfg)
    dm.setup(stage="test")
    dataloader = dm.test_dataloader()
        
    model = model.to(device)
    enable_mc_dropout(model)
    
    all_probs_stack = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            batch_probs = []
            for _ in range(n_passes):
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                batch_probs.append(probs.cpu().numpy())
            # batch_probs: list of N (B, C) -> stack to (N, B, C)
            stacked = np.stack(batch_probs, axis=0)
            all_probs_stack.append(stacked)
            
    # Combine all batches
    # all_probs_stack is list of (N, B, C). We concatenate along B axis (axis 1)
    final_probs_stack = np.concatenate(all_probs_stack, axis=1) # (N, Total_B, C)
    
    # Compute metrics
    probs_tensor = torch.tensor(final_probs_stack)
    pred_ent, mean_ent, mi = compute_mutual_information(probs_tensor)
    
    results_dir = Path("research/baseline_v2/results/uncertainty")
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / f"mc_probs_{split}.npz"
    np.savez(
        out_file, 
        mc_probabilities=final_probs_stack,
        predictive_entropy=pred_ent.numpy(),
        mean_entropy=mean_ent.numpy(),
        mutual_information=mi.numpy()
    )
    logger.info(f"Real MC probabilities and metrics saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--passes", type=int, default=30)
    args = parser.parse_args()
    evaluate_uncertainty(args.checkpoint, args.split, args.passes)



