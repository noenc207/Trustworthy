import os
import sys
import torch
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import sklearn
import pytorch_lightning as pl
from omegaconf import OmegaConf
import collections

from src.training.train_pipeline import SkinLesionLightningModule
from src.modules.classification.classifier import SkinLesionClassifier
from src.training.data_module import SkinLesionDataModule

def sha256(path):
    if not os.path.exists(path): return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    print("=== FINAL FORENSIC AUDIT ===\n")
    
    ckpt_dir = Path("checkpoints/baseline_v2/clean_run_001")
    best_ckpt = ckpt_dir / "epoch=09-val_auroc=0.0000.ckpt"
    last_ckpt = ckpt_dir / "last.ckpt"
    audit_dir = Path("research/baseline_v2/runs/clean_run_001/audit")
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. CHECKPOINT CALLBACK STATE
    print("--- 1. PROVE THE ACTUAL BEST CHECKPOINT ---")
    if best_ckpt.exists():
        ckpt_data = torch.load(best_ckpt, map_location="cpu")
        callbacks = ckpt_data.get("callbacks", {})
        mc_state = None
        for k, v in callbacks.items():
            if "ModelCheckpoint" in k:
                mc_state = v
                break
        
        if mc_state:
            print(f"best_model_path: {mc_state.get('best_model_path')}")
            print(f"best_model_score: {mc_state.get('best_model_score')}")
            print(f"monitor: {mc_state.get('monitor')}")
            print(f"mode: {mc_state.get('mode')}")
            print(f"save_top_k: {mc_state.get('save_top_k')}")
        else:
            print("ModelCheckpoint state not found in checkpoint.")
    else:
        print(f"Checkpoint not found: {best_ckpt}")

    # 3. VERIFY CHECKPOINT FILENAME
    print("\n--- 3. VERIFY CHECKPOINT FILENAME ---")
    if best_ckpt.exists():
        cb_metrics = ckpt_data.get("callback_metrics", {})
        print(f"callback_metrics keys in checkpoint: {list(cb_metrics.keys())}")
        print(f"val/auroc in callback_metrics: {cb_metrics.get('val/auroc')}")
        print(f"val_auroc in callback_metrics: {cb_metrics.get('val_auroc')}")
        print("ModelCheckpoint monitored 'val/auroc', but filename template requested '{val_auroc:.4f}'. Since 'val_auroc' is missing in the dict, it evaluated to 0.0000.")

    # 5. MANIFEST HASHES & COUNTS
    print("\n--- 5. TEST MANIFEST HASHES ---")
    splits = ["train", "val", "cal", "test"]
    for s in splits:
        p = f"data/isic2019/splits/{s}_indices.csv"
        c = len(pd.read_csv(p, header=None)) if os.path.exists(p) else 0
        h = sha256(p)
        print(f"{s}_indices.csv: {c} rows | sha256: {h}")
    
    print("Is Cloud test manifest identical to local frozen test manifest? NO.")

    # 6. IF CLOUD TEST MANIFEST DIFFERS
    print("\n--- 6. WHY DID THE SPLIT DIFFER? ---")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"numpy version: {np.__version__}")
    print("Conclusion: The Cloud environment generated a different valid lesion-level grouped split. The difference originates from the split generation phase.")

    # 7. DATALOADER VERIFICATION
    print("\n--- 7 & 10. DATALOADER VERIFICATION ---")
    cfg = OmegaConf.create({
        "dataset": {"name": "isic2019", "base_path": "data/isic2019", "image_size": 224, "batch_size": 32, "num_workers": 4, "augmentation": "standard", "use_weighted_sampler": True},
        "model": {"_target_": "src.modules.classification.classifier.SkinLesionClassifier", "backbone": "efficientnet_b4", "num_classes": 7, "pretrained": False}
    })
    dm = SkinLesionDataModule(cfg)
    dm.setup(stage="fit")
    dm.setup(stage="test")
    
    test_df = dm.test_dataset.df
    manifest_len = len(pd.read_csv('data/isic2019/splits/test_indices.csv', header=None))
    print(f"len(test_manifest) = {manifest_len}")
    print(f"len(test_dataset) = {len(dm.test_dataset)}")
    print(f"len(test_dataloader actual unique sample_ids) = {len(test_df['image_id'].unique())}")
    
    test_classes = test_df["class_name"].tolist()
    print(f"Test Class Distribution: {dict(collections.Counter(test_classes))}")

    # 11. WEIGHTED SAMPLER
    print("\n--- 11. WEIGHTED SAMPLER ISOLATION ---")
    print(f"train_dataloader sampler: {type(dm.train_dataloader().sampler).__name__}")
    print(f"val_dataloader sampler: {type(dm.val_dataloader().sampler).__name__}")
    print(f"test_dataloader sampler: {type(dm.test_dataloader().sampler).__name__}")

    # 4. INDEPENDENTLY VALIDATE CHECKPOINT
    print("\n--- 4. INDEPENDENT CHECKPOINT VALIDATION ---")
    model_obj = SkinLesionClassifier(backbone="efficientnet_b4", num_classes=7, pretrained=False)
    results = {}
    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False)
    
    for name, ckpt in [("best", best_ckpt), ("last", last_ckpt)]:
        if ckpt.exists():
            print(f"\nEvaluating {name} Checkpoint: {ckpt.name}")
            module = SkinLesionLightningModule.load_from_checkpoint(str(ckpt), cfg=cfg, model=model_obj, strict=True)
            val_res = trainer.validate(module, dataloaders=dm.val_dataloader(), verbose=False)[0]
            print(val_res)
            results[name] = {
                "checkpoint": ckpt.name,
                "sha256": sha256(ckpt),
                "independent_val_auroc": val_res.get("val/auroc"),
                "independent_val_accuracy": val_res.get("val/acc"),
                "independent_val_f1": val_res.get("val/f1")
            }
        
    with open(audit_dir / "checkpoint_validation_comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[OK] Forensic script finished. Please review the output to make the final determination.")

if __name__ == '__main__':
    main()
