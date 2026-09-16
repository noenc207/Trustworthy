"""A0 Reproduction Gate — 10-sample test with fixed canonical preprocessing."""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import numpy as np, torch, cv2
from pathlib import Path
from omegaconf import OmegaConf
from src.training.augmentation import get_val_transforms
from src.training.train_pipeline import SkinLesionLightningModule
from src.modules.classification.classifier import SkinLesionClassifier

IMAGE_SIZE = 224
CKPT = "checkpoints/baseline_v2/clean_run_001/epoch=09-val_auroc=0.0000.ckpt"
IMAGE_DIR = Path("data/isic2019/raw")
tf = get_val_transforms(IMAGE_SIZE)

def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return tf(image=img)["image"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = OmegaConf.create({
    "dataset": {"name": "isic2019", "base_path": "data/isic2019",
                "image_size": 224, "batch_size": 64, "num_workers": 0},
    "model": {"_target_": "src.modules.classification.classifier.SkinLesionClassifier",
              "backbone": "efficientnet_b4", "num_classes": 7, "pretrained": False}
})
backbone = SkinLesionClassifier(backbone="efficientnet_b4", num_classes=7, pretrained=False)
module = SkinLesionLightningModule.load_from_checkpoint(
    CKPT, cfg=cfg, model=backbone, strict=True, weights_only=False, map_location=device)
model = module.model.to(device).eval()

canon = np.load("research/baseline_v2/runs/clean_run_001/predictions/test_canonical.npz", allow_pickle=True)
canon_ids    = canon["image_id"].tolist()
canon_logits = canon["logits"]

rng = np.random.RandomState(42)
sample = rng.choice(len(canon_ids), 10, replace=False)

print("=== A0 REPRODUCTION GATE (10 samples) ===\n")
max_diff = 0.0
agree = 0
for idx in sample:
    iid = canon_ids[idx]
    img = cv2.imread(str(IMAGE_DIR / f"{iid}.jpg"))
    t = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(t).cpu().numpy()[0]
    diff = float(np.abs(logit - canon_logits[idx]).max())
    pred_match = (logit.argmax() == canon_logits[idx].argmax())
    if diff > max_diff:
        max_diff = diff
    if pred_match:
        agree += 1
    print(f"  {iid}: max_logit_diff={diff:.6f}  pred_match={pred_match}")

print(f"\nMax logit diff:        {max_diff:.6f}")
print(f"Prediction agreement:  {agree}/10")
PASS = max_diff <= 1e-4 and agree == 10
print(f"\n=== A0 SMALL-SAMPLE GATE: {'PASS' if PASS else 'FAIL'} ===")
