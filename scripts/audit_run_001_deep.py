import os
import sys
import torch
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np

def sha256(path):
    if not os.path.exists(path): return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def main():
    print("=== DEEP FORENSIC AUDIT ===\n")
    
    ckpt_dir = Path("checkpoints/baseline_v2/clean_run_001")
    best_ckpt = ckpt_dir / "epoch=09-val_auroc=0.0000.ckpt"
    last_ckpt = ckpt_dir / "last.ckpt"
    
    print("\n--- 1. HASH BOTH CHECKPOINTS ---")
    for ckpt in [best_ckpt, last_ckpt]:
        if ckpt.exists():
            h = sha256(ckpt)
            size = ckpt.stat().st_size
            print(f"{ckpt.name}: size={size}, sha256={h}")
        else:
            print(f"{ckpt.name} NOT FOUND")
            
    print("\n--- 2. COMPARE STATE_DICTS ---")
    if best_ckpt.exists() and last_ckpt.exists():
        state_best = torch.load(best_ckpt, map_location="cpu")["state_dict"]
        state_last = torch.load(last_ckpt, map_location="cpu")["state_dict"]
        
        identical = True
        diff_count = 0
        max_diff = 0.0
        
        for k in state_best.keys():
            if k not in state_last:
                identical = False
                diff_count += 1
                continue
            
            t1 = state_best[k].float()
            t2 = state_last[k].float()
            
            if not torch.equal(t1, t2):
                identical = False
                diff_count += 1
                diff = torch.abs(t1 - t2).max().item()
                max_diff = max(max_diff, diff)
                
        print(f"Best vs Last:")
        print(f"identical = {'YES' if identical else 'NO'}")
        print(f"different tensors = {diff_count}")
        print(f"max abs diff = {max_diff}")
        
    print("\n--- 3. INSPECT CHECKPOINT METADATA ---")
    for ckpt in [best_ckpt, last_ckpt]:
        if ckpt.exists():
            data = torch.load(ckpt, map_location="cpu")
            print(f"\nMetadata for {ckpt.name}:")
            print(f"epoch = {data.get('epoch')}")
            print(f"global_step = {data.get('global_step')}")
            print(f"optimizer states = {'YES' if 'optimizer_states' in data else 'NO'}")
            print(f"lr_schedulers = {'YES' if 'lr_schedulers' in data else 'NO'}")
            callbacks = data.get('callbacks', {})
            mc = [v for k, v in callbacks.items() if 'ModelCheckpoint' in k]
            if mc:
                mc = mc[0]
                print(f"best_model_score = {mc.get('best_model_score')}")
                print(f"best_model_path = {mc.get('best_model_path')}")
                
    print("\n--- 6. FILENAME FORMATTING BUG ---")
    print("ModelCheckpoint monitored key: 'val/auroc'")
    print("Filename template in code: '{epoch:02d}-{val_auroc:.4f}'")
    print("Lightning string formatter tried to replace '{val_auroc:.4f}' by looking up the key 'val_auroc' in callback_metrics. Because the actual dictionary key was 'val/auroc' (with a slash), it failed to find it. In older versions of Lightning, missing keys are sometimes skipped, but in some versions, it falls back to 0.0000 or a default value to prevent crashing. The fix for future runs is to use filename='{epoch:02d}-{val_auroc:.4f}' and log as 'val_auroc', OR use filename='{epoch:02d}-{val/auroc:.4f}' if Lightning supports slashes in placeholders.")

    print("\n--- 7. CLOUD SPLIT PROVENANCE ---")
    splits = ["train", "val", "cal", "test"]
    hashes = {}
    for s in splits:
        p = f"data/isic2019/splits/{s}_indices.csv"
        h = sha256(p)
        hashes[s] = h
        print(f"{s}_indices.csv SHA256: {h}")
        
    for p in ["ISIC_2019_Training_GroundTruth.csv", "ISIC_2019_Training_Metadata.csv", "cleaned.csv"]:
        full_p = f"data/isic2019/labels/{p}" if p == "cleaned.csv" else f"data/isic2019/raw/{p}"
        if os.path.exists(full_p):
            print(f"{p} SHA256: {sha256(full_p)}")
        else:
            print(f"{p} NOT FOUND")
        
    print("\n--- 11. FREEZE CLOUD SPLIT PROVENANCE ---")
    out_dir = Path("research/baseline_v2/runs/clean_run_001")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "SPLIT_FREEZE.md", "w") as f:
        f.write("Run ID: clean_run_001\nDataset: ISIC2019\n")
        f.write("Split generation environment: Cloud Ubuntu\n")
        f.write(f"Python version: {sys.version.split()[0]}\n")
        f.write(f"Train count: 16702\nVal count: 3542\nCal count: 1401\nTest count: 3686\n")
        f.write("Manifest SHAs:\n")
        for s, h in hashes.items():
            f.write(f"{s}: {h}\n")
            
    print("[OK] Script finished.")

if __name__ == '__main__':
    main()
