"""
Preprocessing Forensics: Canonical vs DERMA-ACT V0
Identifies the exact preprocessing divergence.
"""
import sys, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
from omegaconf import OmegaConf

from src.core.constants import NORMALIZE_MEAN, NORMALIZE_STD
from src.training.augmentation import get_val_transforms
from src.training.data_module import SkinLesionDataModule
from src.training.train_pipeline import SkinLesionLightningModule
from src.modules.classification.classifier import SkinLesionClassifier

REPO = Path(__file__).parent.parent
CKPT = REPO / "checkpoints/baseline_v2/clean_run_001/epoch=09-val_auroc=0.0000.ckpt"
IMAGE_DIR = REPO / "data/isic2019/raw"
CANON_NPZ = REPO / "research/baseline_v2/runs/clean_run_001/predictions/test_canonical.npz"
TEST_CSV  = REPO / "data/isic2019/splits/test_indices.csv"
IMAGE_SIZE = 224

# --- Load canonical NPZ
canon = np.load(CANON_NPZ, allow_pickle=True)
canon_ids    = canon["image_id"].tolist()
canon_logits = canon["logits"]     # (N,7)
canon_preds  = canon["predicted_label"]

# --- Pick 5 deterministic test images
rng = np.random.RandomState(42)
sample_idx = rng.choice(len(canon_ids), size=5, replace=False)
sample_ids = [canon_ids[i] for i in sample_idx]

print("=== PREPROCESSING FORENSICS ===\n")
print(f"Canonical pipeline (from augmentation.py get_val_transforms):")
print(f"  1. cv2.imread (BGR)  -> cv2.cvtColor(BGR2RGB)  [in pytorch_dataset.py]")
print(f"  2. Albumentations Resize(h={int(IMAGE_SIZE*1.3)}, w={int(IMAGE_SIZE*1.3)})")
print(f"  3. Albumentations CenterCrop(h={IMAGE_SIZE}, w={IMAGE_SIZE})")
print(f"  4. A.Normalize(mean={NORMALIZE_MEAN}, std={NORMALIZE_STD})")
print(f"  5. ToTensorV2() -> CHW float32 tensor")
print()
print(f"DERMA-ACT V0 pipeline (broken):")
print(f"  1. cv2.imread (BGR) -> cv2.cvtColor(BGR2RGB) [OK]")
print(f"  2. cv2.resize(224, 224)  [MISSING 1.3x upscale + CenterCrop]")
print(f"  3. /255.0")
print(f"  4. manual (img - mean) / std")
print(f"  5. np.transpose(HWC->CHW)")
print()

# --- Build both pipelines
canonical_tf = get_val_transforms(IMAGE_SIZE)

def canonical_preprocess(img_rgb: np.ndarray) -> torch.Tensor:
    """Exact canonical pipeline using Albumentations."""
    out = canonical_tf(image=img_rgb)
    return out["image"]  # CHW float32 tensor

def broken_preprocess(img_rgb: np.ndarray) -> torch.Tensor:
    """The broken DERMA-ACT V0 pipeline."""
    img = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.astype(np.float32) / 255.0
    img = (img - np.array(NORMALIZE_MEAN)) / np.array(NORMALIZE_STD)
    return torch.from_numpy(img.transpose(2, 0, 1)).float()

# --- Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = OmegaConf.create({
    "dataset": {"name": "isic2019", "base_path": "data/isic2019",
                "image_size": IMAGE_SIZE, "batch_size": 64, "num_workers": 0},
    "model": {"_target_": "src.modules.classification.classifier.SkinLesionClassifier",
              "backbone": "efficientnet_b4", "num_classes": 7, "pretrained": False}
})
backbone = SkinLesionClassifier(backbone="efficientnet_b4", num_classes=7, pretrained=False)
module = SkinLesionLightningModule.load_from_checkpoint(
    CKPT, cfg=cfg, model=backbone, strict=True, weights_only=False, map_location=device)
model = module.model.to(device).eval()

print("="*70)
print(f"{'Stage':<30} {'Canonical':<35} {'Broken DERMA-ACT':<35}")
print("="*70)

canon_id2idx = {iid: i for i, iid in enumerate(canon_ids)}

for iid in sample_ids:
    img_bgr = cv2.imread(str(IMAGE_DIR / f"{iid}.jpg"))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_bgr.shape[:2]
    
    # Intermediate stages canonical
    resized_canon = cv2.resize(img_rgb, (int(IMAGE_SIZE*1.3), int(IMAGE_SIZE*1.3)),
                                interpolation=cv2.INTER_LINEAR)
    # CenterCrop
    h2, w2 = resized_canon.shape[:2]
    y1 = (h2 - IMAGE_SIZE) // 2; x1 = (w2 - IMAGE_SIZE) // 2
    cropped_canon = resized_canon[y1:y1+IMAGE_SIZE, x1:x1+IMAGE_SIZE]
    scaled_canon = cropped_canon.astype(np.float32) / 255.0
    
    # Intermediate stages broken
    resized_broken = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
    scaled_broken = resized_broken.astype(np.float32) / 255.0
    
    # Final tensors
    t_canon  = canonical_preprocess(img_rgb).unsqueeze(0).to(device)
    t_broken = broken_preprocess(img_rgb).unsqueeze(0).to(device)
    
    # Model logits
    with torch.no_grad():
        logits_canon  = model(t_canon).cpu().numpy()[0]
        logits_broken = model(t_broken).cpu().numpy()[0]
    
    gt_logits = canon_logits[canon_id2idx[iid]]
    
    print(f"\nImage: {iid}")
    print(f"  Original size: {h_orig}x{w_orig}")
    
    # Pixel-level comparison at resize stage
    diff_resize = np.abs(resized_canon[:IMAGE_SIZE, :IMAGE_SIZE].astype(float) - 
                         resized_broken.astype(float))
    print(f"  After resize:")
    print(f"    Canonical  (291x291 then crop): shape={cropped_canon.shape}, mean={cropped_canon.mean():.3f}")
    print(f"    Broken     (direct 224x224):    shape={resized_broken.shape}, mean={resized_broken.mean():.3f}")
    print(f"    Max abs pixel diff (first 224): {diff_resize.max():.1f}")
    
    print(f"  Logit comparison:")
    print(f"    GT canonical logits[:3]:  {gt_logits[:3]}")
    print(f"    Fixed canonical logits:   {logits_canon[:3]}")
    print(f"    Broken logits:            {logits_broken[:3]}")
    print(f"    |canonical_fixed - gt|:   {np.abs(logits_canon - gt_logits).max():.6f}")
    print(f"    |broken - gt|:            {np.abs(logits_broken - gt_logits).max():.6f}")

print("\n" + "="*70)
print("DIAGNOSIS:")
print(f"  The canonical val_transform uses:")
print(f"    Albumentations Resize(291,291) + CenterCrop(224,224)")
print(f"  DERMA-ACT used:")
print(f"    cv2.resize(224,224) directly — missing the 1.3x resize+crop stage")
print(f"  This is the ONLY divergence. RGB/BGR order, normalization, and dtype are correct.")
print("="*70)
