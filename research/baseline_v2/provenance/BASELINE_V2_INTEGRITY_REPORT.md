# BASELINE V2 INTEGRITY AUDIT REPORT

## OVERALL STATUS
**FAIL**

## 1. Dataset
- **Name**: ISIC 2019
- **Total Samples**: 25,331
- **Status**: Local machine only has metadata; actual 9GB image dataset is missing locally, preventing raw pixel recomputations.

## 2. Split Provenance
- **Status**: **FAILED**.
- **Issue**: A severe bug in src/training/data_module.py caused split generation to be skipped on the cloud if cleaned.csv already existed (due to a prior crash). As a result, 	rain_indices.csv and 	est_indices.csv were never generated on the cloud.
- **Consequence**: PyTorch Lightning defaults to loading the entire dataset (25,331 samples) if the split files are missing. The model was trained, validated, calibrated, and tested on **100% overlapping data**.

## 3. Checkpoint Provenance
- **File**: epoch=38-val_auroc=0.0000.ckpt
- **Status**: **SUSPECT**.
- **Reason**: The model architecture and state dictionary load perfectly, but the checkpoint weights are compromised by training on the test/calibration sets (Data Leakage). 

## 4. Validation Recomputation
- **Status**: **IMPOSSIBLE**. 
- **Reason**: The raw ISIC 2019 images are not available on this local machine to re-run the evaluate_baseline_v2.py pipeline. Even if they were, the metrics would be meaningless due to the data leakage in the training phase.

## 5. Test Recomputation
- **Status**: **INVALID ARTIFACTS DETECTED**. 
- **Reason**: Inspection of predictions_baseline_v2_test.npz revealed it contains exactly 25,331 predictions. The true test split for this dataset (using our generated split manifest) is 3,890. This proves the test evaluation was run on the entire training set.

## 6. Calibration Verification
- **Status**: **INVALID**. The calibration predictions were either old smoke artifacts (from 4:13 PM pre-cloud tests) or computed on the entire dataset.

## 7. Selective Prediction Verification
- **Status**: **INVALID**. Based on leaked predictions.

## 8. Conformal Verification
- **Status**: **INVALID**. Based on leaked predictions.

## 9. MC Dropout Verification
- **Status**: **INVALID**. mc_probs_test.npz contains 25,331 entries evaluated 30 times, which is computationally wasteful and scientifically invalid.

## 10. OOD Status
- **Status**: **NOT_AVAILABLE**.
- **Reason**: No externally validated OOD benchmark is currently available in the repository.

## 11. Class-Order Audit
- **Status**: VERIFIED. The class order mapping is consistent across constants.py and isic2019.py.

## 12. Normalization Audit
- **Status**: VERIFIED. Normalization logic correctly uses IMAGE_MEAN and IMAGE_STD.

## 13. Leakage Audit
- **Status**: **CATASTROPHIC LEAKAGE**. Overlap between Train, Val, Calibration, and Test is 100%.

## 14. Reproducibility Audit
- **Status**: N/A (Cannot rerun inference without images).

## 15. Invalid/Suspect Artifacts
- All artifacts in esearch/baseline_v2/results/ have been quarantined to esearch/baseline_v2/suspect_artifacts/audit_2026/.

## 16. Final Trustworthy Artifacts
- **NONE**.

## 17. Remaining Limitations
- A code fix was applied to src/training/data_module.py to ensure splits are strictly generated.
- The model must be **fully retrained** from scratch on the cloud using the fixed split logic.

---
**CONCLUSION**: DO NOT publish these metrics. The reported Accuracy (91.24%) and Conformal Coverage (97.98%) are entirely a result of testing on the training data. Re-training is strictly required.
