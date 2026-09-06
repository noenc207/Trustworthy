"""
Evaluate Baseline V2 predictive performance and save raw predictions.
"""
import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import json
import torch
original_load = torch.load
def safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load
from omegaconf import DictConfig, ListConfig, OmegaConf
import omegaconf.base
import omegaconf.nodes

try:
    from torch.serialization import add_safe_globals
    import omegaconf.base; import omegaconf.nodes; import typing; add_safe_globals([DictConfig, ListConfig, omegaconf.base.ContainerMetadata, omegaconf.nodes.AnyNode, typing.Any])
except ImportError:
    pass

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score, average_precision_score, 
    matthews_corrcoef, cohen_kappa_score, confusion_matrix
)

from src.training.train_pipeline import SkinLesionLightningModule
from src.training.data_module import SkinLesionDataModule

def compute_metrics(y_true, y_pred, y_probs, num_classes=7):
    # Check present classes
    present_classes = np.unique(y_true)
    all_classes_present = len(present_classes) == num_classes
    
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0) # Same as balanced accuracy
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    metrics["sensitivity"] = {}
    metrics["specificity"] = {}
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fp + fn)
        
        metrics["sensitivity"][str(i)] = tp / (tp + fn) if (tp + fn) > 0 else None
        metrics["specificity"][str(i)] = tn / (tn + fp) if (tn + fp) > 0 else None
        
    # AUROC / AUPRC
    if all_classes_present:
        metrics["auroc_macro"] = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
        # One-hot encode for AUPRC
        y_true_onehot = np.zeros((len(y_true), num_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1
        metrics["auprc_macro"] = average_precision_score(y_true_onehot, y_probs, average="macro")
    else:
        logger.warning(f"Only {len(present_classes)}/{num_classes} classes present. AUROC/AUPRC unavailable for full 7-classes.")
        metrics["auroc_macro"] = None
        metrics["auprc_macro"] = None
        
    return metrics, cm


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def evaluate(checkpoint_path: str, split: str = "test"):
    logger.info(f"Evaluating Baseline V2 using checkpoint: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        model_module = SkinLesionLightningModule.load_from_checkpoint(checkpoint_path, strict=True)
        model = model_module.model
        
        # 9. EVALUATION CONFIG MUST COME FROM CHECKPOINT
        cfg = model_module.hparams.cfg if hasattr(model_module.hparams, "cfg") else model_module.hparams
        
    except Exception as e:
        logger.error(f"Failed to load checkpoint strictly: {e}")
        raise RuntimeError(f"Checkpoint load failed: {e}")
        
    # We must use actual dataset paths from the config
    logger.info(f"Loaded config from checkpoint. Dataset: {cfg.dataset.base_path}")
    
    dm = SkinLesionDataModule(cfg)
    dm.setup(stage="test" if split == "test" else "fit")
    
    if split == "test":
        dataloader = dm.test_dataloader()
        dataset = dm.test_dataset
    elif split == "calibration":
        dataloader = dm.val_dataloader()
        dataset = dm.val_dataset
    else:
        raise ValueError(f"Unknown split: {split}")
        
    logger.info(f"Loaded {len(dataset)} samples for {split} split.")
        
    model = model.to(device)
    model.eval()
    
    all_image_ids = []
    all_labels = []
    all_logits = []
    all_probs = []
    all_preds = []
    all_confs = []
    
    dataset.return_metadata = True
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            x, y, metadata = batch
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            
            all_labels.extend(y.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(probs.argmax(dim=-1).cpu().numpy())
            all_confs.extend(probs.max(dim=-1).values.cpu().numpy())
            all_image_ids.extend(metadata["image_id"])

    out_dir = Path("research/baseline_v2/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"predictions_baseline_v2_{split}.npz"
    
    assert len(all_labels) == len(all_logits) == len(all_probs) == len(all_image_ids), "Length mismatch!"
    
    np.savez(
        out_file,
        image_id=np.array(all_image_ids),
        true_label=np.array(all_labels),
        predicted_label=np.array(all_preds),
        logits=np.array(all_logits),
        probabilities=np.array(all_probs),
        confidence=np.array(all_confs)
    )
    logger.info(f"Raw predictions saved to {out_file}")
    
    metrics, cm = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs), num_classes=7)
    
    np.savetxt(out_dir / f"confusion_matrix_{split}.csv", cm, delimiter=",", fmt="%d")
    with open(out_dir / f"metrics_{split}.json", "w") as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)
        
    logger.info(f"Evaluation complete. Accuracy: {metrics['accuracy']:.4f}")
    
    # PHASE 19/30: Overconfidence detection — FIRST-CLASS FAILURE CASE
    confs = np.array(all_confs)
    labels_arr = np.array(all_labels)
    preds_arr = np.array(all_preds)
    wrong = labels_arr != preds_arr
    
    overconf_90 = np.sum((confs >= 0.90) & wrong)
    overconf_95 = np.sum((confs >= 0.95) & wrong)
    total_wrong = np.sum(wrong)
    
    metrics["overconfidence"] = {
        "confidence_ge_0.90_and_wrong": int(overconf_90),
        "confidence_ge_0.95_and_wrong": int(overconf_95),
        "total_wrong": int(total_wrong),
        "total_samples": int(len(labels_arr))
    }
    
    if overconf_90 > 0:
        logger.warning(f"OVERCONFIDENCE ALERT: {overconf_90} predictions with confidence >= 0.90 AND wrong")
    if overconf_95 > 0:
        logger.warning(f"CRITICAL OVERCONFIDENCE: {overconf_95} predictions with confidence >= 0.95 AND wrong")
    
    # Re-save metrics with overconfidence data
    with open(out_dir / f"metrics_{split}.json", "w") as f:
        json.dump(metrics, f, indent=4, cls=NumpyEncoder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints/last.ckpt")
    parser.add_argument("--split", type=str, default="test")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.split)


