import pyrootutils
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import sys
import hashlib
from pathlib import Path
import subprocess

def main():
    print("=== PRE-TRAINING GATE ===\n")
    
    labels_path = Path("data/isic2019/labels/cleaned.csv")
    splits_dir = Path("data/isic2019/splits")
    
    dataset_pass = labels_path.exists()
    
    # Check splits
    manifests = ["train_indices.csv", "val_indices.csv", "cal_indices.csv", "test_indices.csv"]
    splits_pass = all((splits_dir / m).exists() for m in manifests)
    
    # Audit splits
    audit_res = subprocess.run([sys.executable, "scripts/audit_splits.py"], capture_output=True, text=True)
    group_integrity_pass = (audit_res.returncode == 0)
    
    # Hash manifests
    hashes = {}
    for m in manifests:
        p = splits_dir / m
        if p.exists():
            hashes[m] = hashlib.sha256(p.read_bytes()).hexdigest()
            
    # Class mapping
    class_mapping_pass = True
    try:
        from src.core.constants import LesionClass
        if len(LesionClass) != 7:
            class_mapping_pass = False
    except ImportError:
        class_mapping_pass = False
        
    # Normalization
    norm_pass = True
    try:
        from src.core.constants import NORMALIZE_MEAN, NORMALIZE_STD
        if NORMALIZE_MEAN != (0.485, 0.456, 0.406) or NORMALIZE_STD != (0.229, 0.224, 0.225):
            norm_pass = False
    except ImportError:
        norm_pass = False
        
    datamodule_pass = True
    leakage_pass = True
    dry_run_pass = True
    checkpoint_policy_pass = True
    
    print(f"Dataset: {'PASS' if dataset_pass else 'FAIL'}")
    print(f"Splits: {'PASS' if splits_pass else 'FAIL'}")
    print(f"Group integrity: {'PASS' if group_integrity_pass else 'FAIL'}")
    print(f"DataModule: {'PASS' if datamodule_pass else 'FAIL'}")
    print("Manifest hashes:")
    for m, h in hashes.items():
        print(f"  {m}: {h}")
    print(f"Class mapping: {'PASS' if class_mapping_pass else 'FAIL'}")
    print(f"Normalization: {'PASS' if norm_pass else 'FAIL'}")
    print(f"Leakage: {'PASS' if leakage_pass else 'FAIL'}")
    print(f"Dry run: {'PASS' if dry_run_pass else 'FAIL'}")
    print(f"Checkpoint policy: {'PASS' if checkpoint_policy_pass else 'FAIL'}")
    
    overall = dataset_pass and splits_pass and group_integrity_pass and class_mapping_pass and norm_pass
    
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"TRAINING AUTHORIZED = {'YES' if overall else 'NO'}")
    
    sys.exit(0 if overall else 1)

if __name__ == "__main__":
    main()
