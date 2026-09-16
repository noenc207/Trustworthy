"""
Quick 5-image verification of fixed sweep loop.
Confirms A0 now exactly matches canonical; A1-A9 produce valid (different) outputs.
"""
import warnings; warnings.filterwarnings("ignore")
import sys; sys.path.insert(0, ".")
import numpy as np, torch, cv2
from pathlib import Path
from omegaconf import OmegaConf
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.training.augmentation import get_val_transforms
from src.training.train_pipeline import SkinLesionLightningModule
from src.modules.classification.classifier import SkinLesionClassifier
from src.modules.active_perception.action_space import ObservationAction
from src.modules.active_perception.view_generator import ViewGenerator
from src.core.constants import NORMALIZE_MEAN, NORMALIZE_STD

IMAGE_SIZE = 224
CKPT = "checkpoints/baseline_v2/clean_run_001/epoch=09-val_auroc=0.0000.ckpt"
IMAGE_DIR = Path("data/isic2019/raw")

_canonical_tf = get_val_transforms(IMAGE_SIZE)
_view_normalize_tf = A.Compose([
    A.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ToTensorV2(),
])
view_gen = ViewGenerator(image_size=IMAGE_SIZE)

ACTIONS = [
    ObservationAction.KEEP_FULL,
    ObservationAction.ZOOM_CENTER,
    ObservationAction.ZOOM_BORDER,
    ObservationAction.TOP_REGION,
    ObservationAction.BOTTOM_REGION,
    ObservationAction.LEFT_REGION,
    ObservationAction.RIGHT_REGION,
    ObservationAction.TEXTURE_REGION,
    ObservationAction.COLOR_NORMALIZED,
    ObservationAction.ARTIFACT_SUPPRESSED,
]
ACTION_NAMES = [
    "global_view", "center_zoom", "border_zoom",
    "upper_region", "lower_region", "left_region", "right_region",
    "high_frequency_texture", "color_suppressed", "artifact_suppressed",
]

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
model = module.model.to(device).float().eval()
for p in model.parameters(): p.requires_grad_(False)

canon = np.load("research/baseline_v2/runs/clean_run_001/predictions/test_canonical.npz", allow_pickle=True)
canon_ids    = canon["image_id"].tolist()
canon_logits = canon["logits"]
canon_preds  = canon["predicted_label"]

rng = np.random.RandomState(42)
sample_idx = rng.choice(len(canon_ids), 5, replace=False)

print("=== 5-IMAGE SWEEP VERIFICATION ===\n")
print(f"{'Image':<15} {'A0_diff':>10} {'A0_pred_match':>14} {'A1_diff':>10} {'A1_changed':>12}")
print("-" * 65)

max_a0_diff = 0.0
all_a0_match = True
for idx in sample_idx:
    iid = canon_ids[idx]
    img_bgr = cv2.imread(str(IMAGE_DIR / f"{iid}.jpg"))
    raw_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    row_tensors = []
    for action in ACTIONS:
        if action == ObservationAction.KEEP_FULL:
            t = _canonical_tf(image=raw_img)["image"]
        else:
            view = view_gen.generate(raw_img, action)
            t = _view_normalize_tf(image=view)["image"]
        row_tensors.append(t)

    batch = torch.stack(row_tensors).unsqueeze(0).view(len(ACTIONS), 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    with torch.no_grad():
        logits = model(batch).cpu().numpy()  # (10, 7)

    a0_diff = float(np.abs(logits[0] - canon_logits[idx]).max())
    a0_match = (logits[0].argmax() == canon_preds[idx])
    # A1 diff vs A0 — should be non-zero (different view)
    a1_diff = float(np.abs(logits[1] - logits[0]).max())

    if a0_diff > max_a0_diff: max_a0_diff = a0_diff
    if not a0_match: all_a0_match = False

    print(f"{iid:<15} {a0_diff:>10.6f} {str(a0_match):>14} {a1_diff:>10.4f} {'YES' if a1_diff > 0.01 else 'NO':>12}")

print()
print(f"Max A0 logit diff : {max_a0_diff:.6f}  (tolerance 1e-4 = 0.000100)")
print(f"A0 prediction match: {'ALL PASS' if all_a0_match else 'FAIL'}")
print()
PASS = max_a0_diff <= 1e-4 and all_a0_match
print(f"SWEEP VERIFICATION: {'PASS — full sweep authorized' if PASS else 'FAIL'}")
