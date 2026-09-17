# DERMA-ACT V0 — ENTROPY SIGN CORRECTION REPORT

Status: COMPLETE
Date: 2026-09-17

This document logs the critical mathematical correction applied to the entropy metrics in DERMA-ACT V0 artifacts, ensuring the report accurately reflects the mathematical reality of the generated predictions.

---

## 1. The Sign Error

**Previous interpretation**: 
The original analysis exported a column `delta_entropy_mean` which had negative values (e.g., `-0.8518` for `high_frequency_texture`). Due to the negative sign, it was incorrectly assumed this column computed $\Delta H = H_{action} - H_{global}$, leading to the false conclusion that actions *reduced* entropy.

**Root cause**:
The previous script computed $H_{global} - H_{action}$ but labeled it as `delta_entropy_mean`.
Because $H_{global} - H_{action} = -0.8518$, it mathematically requires that $H_{action} > H_{global}$. 

**Correct numerical values (recomputed directly from raw probabilities)**:
*   `global_view` mean entropy = 0.9596
*   `center_zoom` mean entropy = 0.9754
*   `high_frequency_texture` mean entropy = 1.8114

## 2. Correct Formula & Semantics

When computed explicitly as $entropy\_reduction = H_{global} - H_{action}$:
*   `center_zoom` = -0.0158 (Entropy INCREASED by 0.0158)
*   `high_frequency_texture` = -0.8518 (Entropy INCREASED by 0.8518)

Since *all* entropy reduction values were negative, the inescapable mathematical conclusion is:
**All predefined virtual observations increased mean predictive entropy relative to the global view.**

## 3. Impact on V0 Interpretation

1.  **Center Zoom**: Previously mislabeled as having the "highest entropy reduction". It is actually the action with the *smallest mean entropy increase* (+0.0158). It does not reduce entropy on average.
2.  **High-Frequency Texture**: Previously mislabeled as either informative or having high entropy reduction. It actually produces the *largest entropy increase* (+0.8518) relative to global. Combined with its low confidence (0.2748), poor accuracy (0.6845), and massive prediction flip rate (89.7%), it is now correctly described neutrally as a **highly destabilizing virtual transformation** according to the evaluated prediction-stability metrics.
3.  **Predictive Entropy as a Utility Proxy**: Because all virtual observations increased mean predictive entropy, yet several improved classification accuracy, the results indicate that predictive entropy alone is not a reliable proxy for observation utility in this experiment.

## 4. Oracle Semantics Adjusted

Because the oracle relies on minimum entropy over all candidate actions per image, and it successfully improves classification risk, the terminology was sharpened:
*   **Previous**: `model-based offline action-selection oracle`
*   **Corrected**: `minimum-entropy offline action-selection oracle`

This explicitly states the exact criterion ($a^* = \arg\min_a H(p_a)$) without falsely claiming it relies on an "information gain" metric. 
It was also verified that **no test labels, accuracy, or correctness metrics are used** by this oracle for action selection; it relies strictly on model predictive entropy.