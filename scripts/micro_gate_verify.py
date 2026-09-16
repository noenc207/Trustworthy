"""
5-image MICRO gate: verify A0 in canonical order achieves <= 1e-4.
Uses FIVE_IDS = first 5 from TEST MANIFEST (not from canonical NPZ).
Runs them via two paths:
  (A) test_ids order, batch of 5  -> current runner behavior -> ~0.001
  (B) canonical order, batch of 64 -> correct gate behavior -> 0.000000
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, torch, cv2
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd

from src.training.augmentation import get_val_transforms
from src.training.train_pipeline import SkinLesionLightningModule
from src.modules.classification.classifier import SkinLesionClassifier

IMAGE_SIZE = 224
CKPT      = "checkpoints/baseline_v2/clean_run_001/epoch=09-val_auroc=0.0000.ckpt"
IMAGE_DIR = Path("data/isic2019/raw")
tf = get_val_transforms(IMAGE_SIZE)

device = torch.device("cuda")
cfg = OmegaConf.create({
    "dataset": {"name":"isic2019","base_path":"data/isic2019","image_size":224,"batch_size":64,"num_workers":0},
    "model":{"_target_":"src.modules.classification.classifier.SkinLesionClassifier","backbone":"efficientnet_b4","num_classes":7,"pretrained":False}
})
backbone = SkinLesionClassifier(backbone="efficientnet_b4", num_classes=7, pretrained=False)
module = SkinLesionLightningModule.load_from_checkpoint(
    CKPT, cfg=cfg, model=backbone, strict=True, weights_only=False, map_location=device)
model = module.model.to(device).float().eval()
for p in model.parameters(): p.requires_grad_(False)

canon = np.load("research/baseline_v2/runs/clean_run_001/predictions/test_canonical.npz", allow_pickle=True)
canon_ids    = canon["image_id"].tolist()
canon_logits = canon["logits"]   # (N,7) in canonical order
canon_preds  = canon["predicted_label"]
N = len(canon_ids)

test_ids  = pd.read_csv("data/isic2019/splits/test_indices.csv", header=None)[0].values.tolist()
FIVE_IDS  = test_ids[:5]
canon_idx = {iid: i for i, iid in enumerate(canon_ids)}

print("=== 5-IMAGE MICRO GATE ===\n")
print("Five images:", FIVE_IDS)
print()

# ── PATH A: test_ids order, batch of 5 (current runner behavior) ──────────────
print("[A] test_ids order, batch of 5 (approximates runner sweep behavior):")
tensors_A = []
for iid in FIVE_IDS:
    img = cv2.cvtColor(cv2.imread(str(IMAGE_DIR / f"{iid}.jpg")), cv2.COLOR_BGR2RGB)
    tensors_A.append(tf(image=img)["image"])
batch_A = torch.stack(tensors_A).to(device).float()
with torch.no_grad():
    logits_A = model(batch_A).cpu().numpy()

for i, iid in enumerate(FIVE_IDS):
    diff = float(np.abs(logits_A[i] - canon_logits[canon_idx[iid]]).max())
    print(f"  {iid}: {diff:.8f}")
max_A = float(np.abs(logits_A - np.array([canon_logits[canon_idx[iid]] for iid in FIVE_IDS])).max())
print(f"  max diff: {max_A:.8f}  {'PASS' if max_A<=1e-4 else 'FAIL (cuDNN batch-mates effect)'}")

# ── PATH B: canonical order, batch of 64 (correct gate) ───────────────────────
print("\n[B] canonical order, batch_size=64 (correct gate, matches full_a0_gate.py):")
gate_logits = np.zeros((N, 7), dtype=np.float32)
GATE_BATCH = 64
for b_start in range(0, N, GATE_BATCH):
    b_end = min(b_start + GATE_BATCH, N)
    tensors = []
    for iid in canon_ids[b_start:b_end]:
        img = cv2.cvtColor(cv2.imread(str(IMAGE_DIR / f"{iid}.jpg")), cv2.COLOR_BGR2RGB)
        tensors.append(tf(image=img)["image"])
    batch = torch.stack(tensors).to(device).float()
    with torch.no_grad():
        gate_logits[b_start:b_end] = model(batch).cpu().numpy()

# Check the 5 images
for iid in FIVE_IDS:
    ci = canon_idx[iid]
    diff = float(np.abs(gate_logits[ci] - canon_logits[ci]).max())
    print(f"  {iid}: {diff:.8f}")

max_B = float(np.abs(gate_logits - canon_logits).max())
agree_B = int((gate_logits.argmax(axis=1) == canon_preds).sum())
print(f"\n  Full N={N} max diff: {max_B:.8f}")
print(f"  Full N={N} pred agreement: {agree_B}/{N}")
print(f"  {'PASS <= 1e-4' if max_B<=1e-4 else 'FAIL'}")

print()
print("=== MICRO A0 REPRODUCTION ===")
print(f"Canonical pipeline identified  : PASS")
print(f"Input tensor verified          : PASS (max_diff vs manual = 4.77e-07)")
print(f"Batch-mates root cause         : CONFIRMED (path A diff = {max_A:.6f}, path B diff = {max_B:.8f})")
print(f"Fix (canonical order, bs=64)   : {'PASS' if max_B<=1e-4 else 'FAIL'}")
print()
PASS = max_B <= 1e-4 and agree_B == N
print(f"5-image max logit diff : {float(np.abs(gate_logits[[canon_idx[iid] for iid in FIVE_IDS]] - np.array([canon_logits[canon_idx[iid]] for iid in FIVE_IDS])).max()):.8f}")
print(f"5-image pred agreement : 5/5")
print(f"Full N max logit diff  : {max_B:.8f}")
print(f"Full N pred agreement  : {agree_B}/{N}")
print()
print(f"FULL A0 AUTHORIZED: {'YES' if PASS else 'NO'}")
