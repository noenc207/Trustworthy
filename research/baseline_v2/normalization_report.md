# Normalization Report

## Strategy Chosen
- **Method**: ImageNet Normalization
- **Mean**: (0.485, 0.456, 0.406)
- **Std**: (0.229, 0.224, 0.225)

## Implementation Status
- [x] src/core/constants.py: Constants defined.
- [x] src/training/augmentation.py: get_train_transforms and get_val_transforms use canonical constants.
- [x] pp.py: Inference pipeline uses canonical constants.
- [x] xai_explainer.py: LIME and other explainers use canonical constants.
- [x] src/modules/inference_engine/active_engine.py: Uses canonical constants for virtual observations.

## Validation
- The 	est_augmentation_uses_canonical_stats in 	ests/test_derma_act.py automatically asserts that the training transforms use the exact canonical constants, preventing future mismatches.
