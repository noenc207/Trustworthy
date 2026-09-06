# Baseline V2 Calibration Plan

## Objective
Ensure confidence aligns with empirical probability of correctness.

## Calibration Dataset
- A strictly held-out **Calibration Set** (~6-10% of total data) that was NOT used for training or validation (early stopping/hyperparameter tuning).

## Methods to Evaluate
1. **Uncalibrated**: Raw softmax probabilities.
2. **Temperature Scaling (Primary)**: Optimize single temperature scalar T to minimize NLL on the Calibration Set.
3. **Vector Scaling**: Class-specific temperature.
4. **Isotonic Regression**: Non-parametric calibration (if data size permits).

## Metrics
- Expected Calibration Error (ECE)
- Maximum Calibration Error (MCE)
- Brier Score
- Negative Log Likelihood (NLL)
- Class-wise ECE (especially for MEL, BCC, AKIEC)

## Diagnostics
- Plot Reliability Diagrams (Pre and Post Calibration).
- Calculate Confidence Gap (mean(confidence) - accuracy).
