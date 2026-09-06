# Baseline V2 Scientific Audit

## 1. Data Splitting & Leakage
- **Issue**: The original implementation in ase.py used 	rain_test_split(stratify=y) directly on the image level. Since ISIC datasets contain multiple images of the same lesion/patient, this caused **data leakage** (images of the same lesion appearing in both train and validation/test sets).
- **Fix**: Implemented GroupShuffleSplit in src/modules/dataset_manager/patient_split.py. Updated isic2019.py to extract lesion_id from ISIC_2019_Training_Metadata.csv. The split now enforces strict lesion-level separation across 4 subsets (Train, Val, Calibration, Test).

## 2. Normalization Mismatch
- **Issue**: Training used ISIC-specific stats (0.763, 0.546, 0.570)/(0.141, 0.152, 0.169), while inference (pp.py) and XAI (xai_explainer.py) used ImageNet stats.
- **Fix**: Centralized normalization in src/core/constants.py (NORMALIZE_MEAN, NORMALIZE_STD = ImageNet stats). Updated ugmentation.py, pp.py, and xai_explainer.py to use these constants.

## 3. Calibration Implementation
- **Issue**: 	emp_scaling.pkl was fitted on random noise (
p.random.randn) rather than a valid holdout set. Furthermore, calibration was not systematically applied after training.
- **Fix**: Deleted 	emp_scaling.pkl. Added a dedicated **Calibration Set** (approx 6-10%) in the new data split. Calibration will be performed strictly on this set.

## 4. Checkpoint Loading
- **Issue**: strict=False in pp.py was silently ignoring missing keys. The string replacement k.replace('model.', '') replaced all occurrences instead of just the prefix.
- **Fix**: Fixed prefix stripping and enforced strict=True to catch weight mismatches early.

## 5. Metrics & Overconfidence
- **Issue**: Models were evaluated primarily on Accuracy and AUROC. Overconfidence was not systematically penalized.
- **Fix**: Baseline V2 will implement comprehensive calibration metrics (ECE, MCE, Brier, NLL) and evaluate them via scripts/evaluate_baseline_v2.py.
