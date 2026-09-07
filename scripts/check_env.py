import sys
import torch
import timm
import albumentations as A
import pytorch_lightning as pl
import subprocess
import hashlib
import pandas as pd
from pathlib import Path

print("=== ENVIRONMENT ===")
print(f"Python version: {sys.version.split()[0]}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"timm version: {timm.__version__}")
print(f"Albumentations version: {A.__version__}")
print(f"Lightning version: {pl.__version__}")

try:
    print(f"git commit: {subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()}")
except Exception:
    pass

try:
    with open("requirements-lock.txt", "rb") as f:
        print(f"requirements-lock hash: {hashlib.sha256(f.read()).hexdigest()[:8]}")
except Exception:
    pass

print("\n=== DATASET ===")
data_dir = Path("data/isic2019")
if not data_dir.exists():
    print("ERROR: data/isic2019/ does not exist.")
    sys.exit(1)

csv_path = data_dir / "labels/cleaned.csv"
if not csv_path.exists():
    print(f"ERROR: {csv_path} does not exist.")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"TOTAL IMAGES: {len(df)}")
if "lesion_id" in df.columns:
    print(f"TOTAL GROUPS: {df['lesion_id'].nunique()}")
print(f"CLASS COUNTS:\n{df['class_id'].value_counts().to_string()}")

splits_dir = data_dir / "splits"
if splits_dir.exists():
    for s in ["train", "val", "calibration", "test"]:
        p = splits_dir / f"{s}_indices.csv"
        if p.exists():
            print(f"{s.upper()} COUNT: {len(pd.read_csv(p, header=None))}")
        else:
            print(f"{s.upper()} COUNT: 0 (Missing)")
else:
    print("SPLITS: Missing")
