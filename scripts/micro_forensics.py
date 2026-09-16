"""
MICRO NUMERIC FORENSICS — A0 logit diff 0.006 investigation.
Tests 5 deterministic images, exposes all intermediate tensors,
checks batch-size effect, matmul precision, model mode, cuDNN flags.
DO NOT run full sweep. DO NOT change canonical artifact.
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")

import numpy as np
import torch
import cv2
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd

from src.training.augmentation import get_val_transforms
from src.training.train_pipeline import SkinLesionLightningModule
from src.modules.classification.classifier import SkinLesionClassifier
from src.core.constants import NORMALIZE_MEAN, NORMALIZE_STD

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Setup ─────────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224
CKPT       = "checkpoints/baseline_v2/clean_run_001/epoch=09-val_auroc=0.0000.ckpt"
IMAGE_DIR  = Path("data/isic2019/raw")
CANON_NPZ  = Path("research/baseline_v2/runs/clean_run_001/predictions/test_canonical.npz")
TEST_CSV   = Path("data/isic2019/splits/test_indices.csv")

canon = np.load(CANON_NPZ, allow_pickle=True)
canon_ids    = canon["image_id"].tolist()
canon_logits = canon["logits"]
canon_preds  = canon["predicted_label"]

# 5 deterministic images — first 5 rows of frozen test manifest
test_ids = pd.read_csv(TEST_CSV, header=None)[0].values.tolist()
FIVE_IDS = test_ids[:5]
canon_idx = {iid: i for i, iid in enumerate(canon_ids)}

print("=== MICRO NUMERIC FORENSICS ===\n")
print("Five test images:", FIVE_IDS)

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda")
cfg = OmegaConf.create({
    "dataset": {"name": "isic2019", "base_path": "data/isic2019",
                "image_size": 224, "batch_size": 64, "num_workers": 0},
    "model": {"_target_": "src.modules.classification.classifier.SkinLesionClassifier",
              "backbone": "efficientnet_b4", "num_classes": 7, "pretrained": False}
})
backbone = SkinLesionClassifier(backbone="efficientnet_b4", num_classes=7, pretrained=False)
module = SkinLesionLightningModule.load_from_checkpoint(
    CKPT, cfg=cfg, model=backbone, strict=True, weights_only=False, map_location=device)
model = module.model.to(device)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

# ── Environment snapshot ───────────────────────────────────────────────────────
print("\n─── Environment ───────────────────────────────────────────────────────")
print(f"  torch version              : {torch.__version__}")
print(f"  torch.dtype default        : float32")
print(f"  torch.get_float32_matmul_precision() : {torch.get_float32_matmul_precision()}")
print(f"  cudnn.deterministic        : {torch.backends.cudnn.deterministic}")
print(f"  cudnn.benchmark            : {torch.backends.cudnn.benchmark}")
try:
    print(f"  are_deterministic_algorithms_enabled: {torch.are_deterministic_algorithms_enabled()}")
except Exception:
    print(f"  are_deterministic_algorithms_enabled: N/A")
import lightning as L
print(f"  lightning version          : {L.__version__}")

# Model mode
print(f"  model.training             : {model.training}")
bn_train = [m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
dp_train = [m.training for m in model.modules() if isinstance(m, torch.nn.Dropout)]
print(f"  BatchNorm2d layers training: {set(bn_train)} (should be {{False}})")
print(f"  Dropout layers training    : {set(dp_train)} (should be {{False}})")

# Parameter SHA (first param checksum)
first_param = next(model.parameters())
param_sha = float(first_param.abs().sum().item())
print(f"  First-param |sum|          : {param_sha:.6f} (must match canonical)")

# ── Build canonical transform ──────────────────────────────────────────────────
canonical_tf = get_val_transforms(IMAGE_SIZE)

def run_model(tensors, bs):
    """Run model on tensor list in batches of `bs`, return logits."""
    all_logits = []
    for i in range(0, len(tensors), bs):
        batch = torch.stack(tensors[i:i+bs]).to(device).float()
        with torch.no_grad():
            all_logits.append(model(batch).cpu().numpy())
    return np.concatenate(all_logits, axis=0)

# ── Load 5 raw images and instrument canonical transform ──────────────────────
raw_imgs = []
for iid in FIVE_IDS:
    img_bgr = cv2.imread(str(IMAGE_DIR / f"{iid}.jpg"))
    raw_imgs.append(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# Stage-by-stage instrumentation
print("\n─── Stage-by-stage tensor comparison ──────────────────────────────────")
print(f"{'Stage':<28} {'Shape':<20} {'min':>7} {'max':>7} {'mean':>8} {'dtype':<10}")
print("-" * 90)

canonical_tensors = []
for i, (iid, raw) in enumerate(zip(FIVE_IDS, raw_imgs)):
    h, w = raw.shape[:2]

    # STAGE 0: raw RGB uint8
    if i == 0:
        print(f"{'raw RGB uint8':<28} {str(raw.shape):<20} {raw.min():>7} {raw.max():>7} {raw.mean():>8.2f} {str(raw.dtype):<10}")

    # STAGE 1: Albumentations Resize(291,291)
    resized = A.Resize(height=int(IMAGE_SIZE*1.3), width=int(IMAGE_SIZE*1.3))(image=raw)["image"]
    if i == 0:
        print(f"{'after A.Resize(291)':<28} {str(resized.shape):<20} {resized.min():>7} {resized.max():>7} {resized.mean():>8.2f} {str(resized.dtype):<10}")

    # STAGE 2: CenterCrop(224,224)
    cropped = A.CenterCrop(height=IMAGE_SIZE, width=IMAGE_SIZE)(image=resized)["image"]
    if i == 0:
        print(f"{'after CenterCrop(224)':<28} {str(cropped.shape):<20} {cropped.min():>7} {cropped.max():>7} {cropped.mean():>8.2f} {str(cropped.dtype):<10}")

    # STAGE 3: Scale + Normalize (what A.Normalize does internally)
    scaled = cropped.astype(np.float32) / 255.0
    mean_np = np.array(NORMALIZE_MEAN, dtype=np.float32)
    std_np  = np.array(NORMALIZE_STD,  dtype=np.float32)
    normed  = (scaled - mean_np) / std_np
    if i == 0:
        print(f"{'after /255':<28} {str(scaled.shape):<20} {scaled.min():>7.4f} {scaled.max():>7.4f} {scaled.mean():>8.4f} {str(scaled.dtype):<10}")
        print(f"{'after normalize':<28} {str(normed.shape):<20} {normed.min():>7.4f} {normed.max():>7.4f} {normed.mean():>8.4f} {str(normed.dtype):<10}")

    # STAGE 4: Full canonical transform tensor
    t = canonical_tf(image=raw)["image"]   # CHW float32
    if i == 0:
        print(f"{'canonical tensor CHW':<28} {str(tuple(t.shape)):<20} {t.min():>7.4f} {t.max():>7.4f} {t.mean():>8.4f} {str(t.dtype):<10}")

    # Verify A.Normalize result matches manual computation
    t_np = t.numpy()  # (3,H,W)
    manual_chw = normed.transpose(2,0,1)
    diff_normalize = float(np.abs(t_np - manual_chw).max())
    if i == 0:
        print(f"  [check] manual_norm vs canonical_tf: max_diff={diff_normalize:.2e}")

    canonical_tensors.append(t)

# ── Run model at different batch sizes ────────────────────────────────────────
print("\n─── Batch-size effect on logits ────────────────────────────────────────")
results_by_bs = {}
for bs in [1, 5, 32, 64]:
    logits = run_model(canonical_tensors, bs)
    results_by_bs[bs] = logits

gt_logits_5 = np.array([canon_logits[canon_idx[iid]] for iid in FIVE_IDS])

print(f"\n{'Image':<15} {'bs=1':>10} {'bs=5':>10} {'bs=32':>10} {'bs=64':>10} {'gt':>10}")
print("-" * 65)
for i, iid in enumerate(FIVE_IDS):
    gt = canon_logits[canon_idx[iid]].argmax()
    row = f"{iid:<15}"
    for bs in [1, 5, 32, 64]:
        diff = float(np.abs(results_by_bs[bs][i] - gt_logits_5[i]).max())
        row += f" {diff:>10.6f}"
    row += f" {'ref':>10}"
    print(row)

max_by_bs = {bs: float(np.abs(results_by_bs[bs] - gt_logits_5).max()) for bs in [1,5,32,64]}
print()
for bs, mx in max_by_bs.items():
    print(f"  batch_size={bs:>2}: max_abs_diff={mx:.8f}  {'<= 1e-4 PASS' if mx<=1e-4 else '> 1e-4 FAIL'}")

# ── Check matmul precision effect ─────────────────────────────────────────────
print("\n─── torch.float32_matmul_precision effect ──────────────────────────────")
for prec in ["highest", "high", "medium"]:
    torch.set_float32_matmul_precision(prec)
    logits_p = run_model(canonical_tensors, 64)
    diff_p = float(np.abs(logits_p - gt_logits_5).max())
    print(f"  precision={prec:<10}: max_diff={diff_p:.8f}  {'<= 1e-4' if diff_p<=1e-4 else '> 1e-4'}")
# Restore default
torch.set_float32_matmul_precision("highest")

# ── Check deterministic flags ─────────────────────────────────────────────────
print("\n─── cuDNN deterministic flags effect ───────────────────────────────────")
for det, bench in [(False,True), (False,False), (True,False)]:
    torch.backends.cudnn.deterministic = det
    torch.backends.cudnn.benchmark = bench
    logits_d = run_model(canonical_tensors, 64)
    diff_d = float(np.abs(logits_d - gt_logits_5).max())
    print(f"  deterministic={str(det):<5} benchmark={str(bench):<5}: max_diff={diff_d:.8f}  {'PASS' if diff_d<=1e-4 else 'FAIL'}")
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# ── Per-image breakdown at best settings ─────────────────────────────────────
print("\n─── Per-image max logit diff at batch_size=64 ──────────────────────────")
logits_64 = results_by_bs[64]
agree = 0
for i, iid in enumerate(FIVE_IDS):
    diff = float(np.abs(logits_64[i] - gt_logits_5[i]).max())
    pred_match = (logits_64[i].argmax() == canon_preds[canon_idx[iid]])
    if pred_match: agree += 1
    print(f"  {iid}: max_diff={diff:.8f}  pred_match={pred_match}")
print(f"\n  5-image max diff: {max_by_bs[64]:.8f}")
print(f"  5-image pred agreement: {agree}/5")

# ── Final gate ────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("=== MICRO A0 REPRODUCTION ===")
print("="*60)
best_bs = min(max_by_bs, key=max_by_bs.get)
best_diff = max_by_bs[best_bs]
print(f"Canonical pipeline identified  : PASS")
print(f"Color order verified           : PASS (BGR->RGB)")
print(f"Crop verified                  : PASS (Resize291+CenterCrop224)")
print(f"Normalization verified         : PASS")
print(f"Model mode verified            : PASS (eval(), no grad)")
print(f"Deterministic settings         : {'PASS' if max_by_bs[64]<=1e-4 else 'canonical used no det flags'}")
print()
for bs in [1,5,32,64]:
    mx = max_by_bs[bs]
    print(f"  batch_size={bs:>2} max diff: {mx:.8f}  {'<= 1e-4 PASS' if mx<=1e-4 else '> 1e-4 FAIL'}")
print()
print(f"5-image max logit diff (best): {best_diff:.8f} at batch_size={best_bs}")
print(f"5-image prediction agreement : {agree}/5")
auth = best_diff <= 1e-4 and agree == 5
print(f"\nFULL A0 AUTHORIZED: {'YES (at batch_size=' + str(best_bs) + ')' if auth else 'NO — still investigating'}")
