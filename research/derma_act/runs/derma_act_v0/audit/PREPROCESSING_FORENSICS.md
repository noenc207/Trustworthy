# DERMA-ACT V0 Preprocessing Forensics Report — FINAL

**Status**: RESOLVED (FULL GATE PASS)
**Date**: 2026-09-16

---

## Root Cause

### First Divergence: Resize stage

| Stage | Canonical (get_val_transforms) | Broken DERMA-ACT |
|---|---|---|
| Image loader | cv2.imread | cv2.imread |
| Color order | BGR→RGB | BGR→RGB |
| **Resize** | **A.Resize(291, 291, INTER_LINEAR)** | **cv2.resize(224, 224) ← FIRST DIVERGENCE** |
| **CenterCrop** | **A.CenterCrop(224, 224)** | **None** |
| Scale | /255 (implicit in A.Normalize) | /255 |
| Normalization | ImageNet (0.485,0.456,0.406)/(0.229,0.224,0.225) | Same |
| Tensor layout | CHW float32 | CHW float32 |

The canonical `get_val_transforms()` performs an anti-shortcut-learning
strategy: upsample to 1.3x (291x291) then CenterCrop to 224x224.
This removes ~15% border pixels (hospital watermarks, scale bars, logos).

The DERMA-ACT preprocess() was written from scratch and missed this step.

---

## Canonical Pipeline (Full Call Chain)

```
data/isic2019/raw/<id>.jpg
  └► cv2.imread()                              HWC BGR uint8
  └► cv2.cvtColor(BGR2RGB)                     HWC RGB uint8
  └► A.Resize(h=291, w=291, INTER_LINEAR)      HWC RGB uint8
  └► A.CenterCrop(h=224, w=224)                HWC RGB uint8
  └► A.Normalize(mean=(0.485,0.456,0.406),
                 std=(0.229,0.224,0.225))       HWC RGB float32
  └► ToTensorV2()                              CHW float32 tensor
  └► SkinLesionClassifier.forward()
  └► logits (7,)
```

Source: src/training/augmentation.py:get_val_transforms()

---

## Effect of Bug

| Metric | Broken | Fixed |
|---|---|---|
| Max logit diff vs canonical | 3.522 | 0.000000 |
| Prediction agreement (full N=3686) | 81.3% | 100% |

---

## Fix Applied

scripts/run_derma_act_v0.py now uses:

```python
from src.training.augmentation import get_val_transforms as _get_val_transforms
_canonical_tf = _get_val_transforms(image_size=IMAGE_SIZE)

def preprocess(img_rgb):
    return _canonical_tf(image=img_rgb)["image"]
```

---

## Full Gate Results

| Device | Max logit diff | Pred agreement | PASS? |
|---|---|---|---|
| CPU | 0.016516 | 3684/3686 | FAIL |
| **GPU (CUDA)** | **0.000000** | **3686/3686** | **PASS** |

The canonical test_canonical.npz was generated on GPU (same CUDA device).
CPU fails due to FP32 matmul differences between CPU and GPU kernels.
The runner enforces CUDA is required (HALT if CUDA unavailable).

---

## Environment Versions

| Package | Version |
|---|---|
| torch | 2.5.1+cu121 |
| albumentations | 1.3.1 |
| pytorch_lightning | 2.6.5 (checkpoint from 2.6.6) |

---

## Final Gate

```
=== A0 REPRODUCTION GATE ===

Canonical preprocessing identified : PASS  (get_val_transforms)
Color order verified               : PASS  (BGR->RGB)
Crop verified (Resize291+Crop224)  : PASS  (Albumentations)
Resize verified                    : PASS  (INTER_LINEAR)
Normalization verified             : PASS  (ImageNet mean/std)
Model path verified                : PASS  (strict=True)
Small-sample A0 reproduction       : PASS  (10/10 pred agree)
Full A0 reproduction (GPU)         : PASS

Max logit diff  : 0.000000
Prediction agree: 3686/3686

DERMA-ACT OBSERVATION SWEEP AUTHORIZED: YES
```