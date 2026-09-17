# DERMA-ACT V0 — SEMANTIC CORRECTION REPORT

Status: COMPLETE
Date: 2026-09-17

This document logs the semantic corrections applied to the DERMA-ACT V0 artifacts after the mathematical audit passed.

---

## 1. Classification Risk vs. Confidence Uncertainty

**Problem**: The original V0 report conflated two distinct metrics under the term "risk". One was based on $1 - Accuracy$ (empirical classification error) and the other on $1 - \max p(c|x)$ (the complement of predictive confidence).

**Correction**: 
*   **Classification risk** is strictly defined as $1 - Accuracy$. 
*   The complement of confidence ($1 - \max p(c|x)$) has been explicitly renamed to **confidence_uncertainty_proxy**.
*   The "fraction_captured" metric derived from confidence uncertainty has been removed from primary conclusions to prevent misleading interpretations.

## 2. Full-Observation Decoupling

**Problem**: Previous texts erroneously implied that aggregating all views (full observation control) "increased risk" because the confidence_uncertainty_proxy worsened, while simultaneously reporting that accuracy improved.

**Correction**: We explicitly decoupled the metrics in the report. The correct neutral interpretation is:
> Naive averaging of all virtual views improved classification accuracy (risk drops from 0.3079 to 0.2705) while producing lower predictive confidence under the confidence-complement proxy (proxy increases from 0.3320 to 0.4844).

## 3. High-Frequency Texture Semantics

**Problem**: The texture view produced a `delta_entropy_mean` of `-0.8518` (Convention A: $H_{action} - H_{global}$). Because this was a mathematical reduction in entropy, prior drafts might mistakenly call it "informative", while simultaneously having terrible accuracy and very low confidence.

**Correction**: We replaced value judgments with a neutral description:
> The texture transformation produces a large reduction in predictive entropy according to the frozen classifier, while simultaneously reducing maximum predicted probability, lowering accuracy relative to global view, and causing a very high prediction-flip rate.

This serves as a critical example that uncertainty metrics and decision quality can strongly disagree.

## 4. "Highest Information Gain" Claim

**Problem**: A sorting bug (`ascending=False` on negative values) led the initial report to declare `center_zoom` as having the highest entropy reduction.

**Correction**: `high_frequency_texture` is mathematically the action with the largest entropy reduction (most negative $\Delta H$). The phrase "best action" is removed, recognizing that entropy reduction does not equal accuracy improvement.

## 5. Terminology Overhaul

Several conceptually flawed terms were corrected:

*   **"ceiling", "oracle", "maximum possible benefit" (when referring to mean of all views)**
    *   *Corrected to*: `full_observation_control`. 
    *   *Reason*: It is a descriptive control of indiscriminate mean aggregation, not a theoretical maximum.
*   **"model_oracle_B1" / "active policy"**
    *   *Corrected to*: `model_based_offline_action_selection_oracle`.
    *   *Reason*: It selects actions by evaluating all candidates beforehand (optimistic, offline). It is not a deployable sequential active policy.
*   **"fixed policies"**
    *   *Corrected to*: `fixed predetermined sequence policies`.
    *   *Reason*: These are not "active" since they do not adapt to model output.

## 6. Redefined Core Finding

The main conclusion was shifted from a generic "more evidence is better" to the more precise, nuanced finding:

> Different virtual observations have heterogeneous and sometimes conflicting effects on classification accuracy, predictive confidence, and prediction stability. An offline model-based action-selection procedure achieved higher accuracy than predefined fixed-view policies in this experiment, suggesting that evidence selection may be more important than indiscriminate evidence aggregation. Observation value is action-dependent and non-monotonic.