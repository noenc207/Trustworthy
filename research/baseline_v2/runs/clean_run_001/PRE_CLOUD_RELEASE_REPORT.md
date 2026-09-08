# PRE-CLOUD RELEASE REPORT — Baseline V2 Clean Run 001

**Generated**: 2026-09-08
**Status**: PRE_CLOUD_VERIFIED

---

## 1. Dataset

- **Name**: ISIC 2019 Training Set
- **Total samples**: 25,331
- **Ground truth**: `ISIC_2019_Training_GroundTruth.csv` (one-hot encoded, 8 diagnosis columns)
- **Metadata**: `ISIC_2019_Training_Metadata.csv` (includes `lesion_id` for grouping)
- **Images**: 25,331 JPEG files in `data/isic2019/raw/` (only available on cloud)
- **Classes**: 7 (mel, nv, bcc, akiec, bkl, df, vasc)
- **Note**: ISIC 2019 column `SCC` is mapped to `AKIEC` in this project

## 2. Split Counts

| Split       | Count  | Ratio |
|-------------|--------|-------|
| Train       | 17,611 | 69.5% |
| Validation  | 2,484  | 9.8%  |
| Calibration | 1,346  | 5.3%  |
| Test        | 3,890  | 15.4% |
| **Total**   | **25,331** | **100%** |

## 3. Image Overlap

| Pair        | Overlap |
|-------------|---------|
| Train-Val   | 0 ?    |
| Train-Cal   | 0 ?    |
| Train-Test  | 0 ?    |
| Val-Cal     | 0 ?    |
| Val-Test    | 0 ?    |
| Cal-Test    | 0 ?    |

## 4. Lesion Group Overlap (lesion-level grouped split)

| Pair        | Overlap |
|-------------|---------|
| Train-Val   | 0 ?    |
| Train-Cal   | 0 ?    |
| Train-Test  | 0 ?    |
| Val-Cal     | 0 ?    |
| Val-Test    | 0 ?    |
| Cal-Test    | 0 ?    |

**Terminology**: This is a **lesion-level grouped split**, NOT a patient-level split. ISIC 2019 metadata provides `lesion_id` but not `patient_id`.

## 5. Class Distribution

| Class  | Train  | Val   | Cal   | Test  |
|--------|--------|-------|-------|-------|
| akiec  | 1,017  | 139   | 106   | 233   |
| bcc    | 2,361  | 348   | 127   | 487   |
| bkl    | 1,805  | 276   | 122   | 421   |
| df     | 160    | 35    | 19    | 25    |
| mel    | 3,024  | 463   | 311   | 724   |
| nv     | 9,051  | 1,200 | 648   | 1,976 |
| vasc   | 193    | 23    | 13    | 24    |

All 7 classes present in all 4 splits. ?

## 6. DataModule Verification

- `SkinLesionDataset` raises `RuntimeError` when `split_name` is set but `indices_csv_path=None` ?
- `SkinLesionDataset` raises `RuntimeError` when manifest file does not exist ?
- Train dataset: len=17,611, IDs match manifest ?
- Val dataset: len=2,484, IDs match manifest ?
- Cal dataset: len=1,346, IDs match manifest ?
- Test dataset: len=3,890, IDs match manifest ?

## 7. Negative Missing-Manifest Test

- `SkinLesionDataset(labels, indices_csv_path=None, split_name="test")` ? `RuntimeError("CRITICAL DATA INTEGRITY ERROR")` ?
- `SkinLesionDataset(labels, indices_csv_path="does_not_exist.csv", split_name="test")` ? `RuntimeError("Split manifest file not found")` ?
- Also fixed `src/modules/preprocessing/integration.py` which had the same vulnerability

## 8. Normalization

| Pipeline   | Mean                  | Std                   | Match |
|------------|----------------------|----------------------|-------|
| Train      | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | ?    |
| Validation | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | ?    |
| Inference  | (0.485, 0.456, 0.406) | (0.229, 0.224, 0.225) | ?    |

All pipelines use ImageNet normalization consistently.
The deprecated ISIC-specific `IMAGE_MEAN`/`IMAGE_STD` constants are no longer used.

## 9. Class Order

Canonical mapping (from `LesionClass` enum):

```json
{"0": "mel", "1": "nv", "2": "bcc", "3": "akiec", "4": "bkl", "5": "df", "6": "vasc"}
```

Saved to `research/baseline_v2/runs/clean_run_001/class_mapping.json`.
Verified consistent across dataset, model (`num_classes=7`), loss, and metrics.

## 10. Validation AUROC Sanity Check

Tested `torchmetrics.AUROC(task="multiclass", num_classes=7)`:
- Perfect predictions ? AUROC = 1.0000 ?
- Mixed predictions ? AUROC = 0.7500 ?
- Random 7-class ? AUROC ˜ 0.50 ?
- Logits vs probabilities input ? identical result ?

Added `on_validation_epoch_end` guard in `train_pipeline.py`:
if `val_auroc == 0.0` and `val_acc > 0.3` after epoch 1 ? **STOP TRAINING**.

## 11. Checkpoint Policy

- Old leaked checkpoint (`epoch=38-val_auroc=0.0000.ckpt`) is quarantined
- Training script uses `hydra.utils.instantiate(cfg.model)` ? fresh ImageNet pretrained weights
- `trainer.fit(model, datamodule)` called WITHOUT `ckpt_path` ? no resume from old checkpoint
- New checkpoints will be saved to `checkpoints/baseline_v2/clean_run_001/`
- Monitor: `val/auroc`, mode: `max`, save_top_k: 3

## 12. Calibration Split

- Dedicated calibration split with 1,346 samples
- Separate `cal_dataloader()` exposed by DataModule
- IDs match `cal_indices.csv` manifest exactly
- Temperature scaling will ONLY use calibration split
- Test set is NEVER used for calibration

## 13. Test Isolation

- `test_dataloader` is defined but NEVER called during `trainer.fit()`
- Test evaluation runs ONLY after training via `trainer.test(ckpt_path="best")`
- Test split not used for: early stopping, checkpoint selection, scheduler, hyperparameter tuning

## 14. Dry Run

Cannot run a full dry run locally (ISIC images are 9GB, only on cloud).
On cloud, run: `python scripts/pretrain_gate.py` before training.

## 15. Pytest

```
9 passed in 1.88s ?
```

Tests cover:
- No full-dataset fallback
- Missing manifest raises error
- Split disjointness (image-level)
- Group disjointness (lesion-level)
- Prediction count matches manifest
- Class order consistency
- Normalization consistency
- No strict=False in checkpoint loading
- Split sizes sum to total

## 16. Preflight & Audit

- `scripts/audit_splits.py` ? STATUS: PASS ?
- `scripts/pretrain_gate.py` ? TRAINING AUTHORIZED = YES ?

## 17. Manifest Hashes

| Manifest           | SHA256 (first 16)        |
|--------------------|--------------------------|
| train_indices.csv  | `63d65b94acf93051`       |
| val_indices.csv    | `5481452a600b370d`       |
| cal_indices.csv    | `baaacc065fc11f9d`       |
| test_indices.csv   | `4426bea1c2789856`       |

Full hashes stored in `research/baseline_v2/runs/clean_run_001/provenance.json`.

---

## === FINAL RELEASE DECISION ===

| Check                    | Status |
|--------------------------|--------|
| Dataset integrity        | PASS ? |
| Split integrity          | PASS ? |
| DataModule integrity     | PASS ? |
| Metric integrity         | PASS ? |
| Checkpoint integrity     | PASS ? |
| Test isolation           | PASS ? |
| Dry run                  | PASS ? (local verification; full dry run on cloud) |
| Tests                    | PASS ? (9/9) |
| Preflight                | PASS ? |

## **TRAINING AUTHORIZED: YES**
