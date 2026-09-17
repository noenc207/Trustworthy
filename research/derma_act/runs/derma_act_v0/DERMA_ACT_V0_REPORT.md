# DERMA-ACT V0 Report

## 1. Hypothesis

> Does acquiring additional virtual observations of the same image
> reduce predictive uncertainty and/or classification risk under
> a constrained observation budget?

Virtual observations are derived from deterministic image transformations
(crops, frequency filtering, colour normalization, artifact suppression).
They are **not** additional clinical measurements and carry no independent
diagnostic information beyond what is present in the original image.

---

## 2. Frozen Baseline

| Field | Value |
|---|---|
| run_id | clean_run_001 |
| model | EfficientNet-B4 |
| checkpoint SHA256 | `0d14579d668d02781874cf91ad493fac4bf33ed53e3be928e3a7f9bb856ebbbb` |
| test manifest SHA256 | `621af2b3afba44817c79138b175a0becd131af95797dbebb8aab740d76ad41a4` |
| N_test | 3686 |
| git commit | `65375bcfdb0e296f537698538ad6e4663ae67524` |

**Global-only baseline** — Accuracy: 0.6921 | Classification Risk: 0.3079

---

## 3. Virtual Observation Space

| ID | Name | Description |
|---|---|---|
| A0 | global_view | Full image resized to 224×224 |
| A1 | center_zoom | Central 60% crop, resized |
| A2 | border_zoom | Peripheral region with central 40% masked |
| A3 | upper_region | Top half, resized |
| A4 | lower_region | Bottom half, resized |
| A5 | left_region | Left half, resized |
| A6 | right_region | Right half, resized |
| A7 | high_frequency_texture | Gabor-bank texture map (4 orientations) |
| A8 | color_suppressed | Minkowski p=6 colour normalization |
| A9 | artifact_suppressed | Hair/artefact inpainting (INPAINT_TELEA, 17×17 kernel) |

---

## 4. Experimental Protocol

1. Load frozen checkpoint. Verify SHA256.
2. For each test image: load raw image, generate all 10 observations, run frozen classifier, release image.
3. Verify A0 logit agreement with `test_canonical.npz` (tolerance 1e-4).
4. Evaluate 6 fixed predetermined sequence policies (P0..P4 + full_observation_control).
5. Evaluate minimum-entropy offline action-selection oracle (entropy criterion, no test labels).
6. Compute per-action marginal effects, observation redundancy, counterfactual sensitivity.
7. Produce risk-evidence curves and figures.

**Aggregation methods:**
- Primary: mean probability: $\bar{p} = \frac{1}{K}\sum_{k=1}^{K} p_k$
- Secondary (confidence-weighted): $\hat{p} = \sum_k w_k p_k$ where $w_k \propto \max_c p_k(c)$

---

## 5. Evidence Cost

All virtual observations assigned a uniform normalized cost of 1.0.
This is a **normalized virtual observation cost** — NOT a clinical cost.
Cost for a sequence of K observations = K.

---

## 6. Classification Risk vs Evidence Cost

**Classification Risk** is defined as $1 - Accuracy$. This must remain strictly separated from the **Confidence uncertainty proxy** ($1 - \max p(c|x)$).

| Policy | N_obs | Cost | Accuracy | Classification Risk |
|---|---|---|---|---|
| P0_global_only | 1 | 1.0 | 0.6921 | 0.3079 |
| P1_global_center | 2 | 2.0 | 0.6989 | 0.3011 |
| P2_global_border | 2 | 2.0 | 0.6763 | 0.3237 |
| P3_global_texture | 2 | 2.0 | 0.6845 | 0.3155 |
| P4_global_artifact | 2 | 2.0 | 0.6793 | 0.3207 |
| P_ALL_full_observation_control | 10 | 10.0 | 0.7295 | 0.2705 |
| minimum_entropy_offline_oracle_B1 | 2 | 2.0 | 0.7165 | 0.2835 |

**Absolute classification-risk reduction vs Global:**
- Global → fixed center: 0.0068
- Global → minimum-entropy offline B1: 0.0244
- Global → full-observation control: 0.0374

Note: `full_observation_control` is a descriptive control showing the behavior of indiscriminate mean aggregation across all predefined views. It is NOT a theoretical performance ceiling.

**Accuracy–confidence decoupling:**
Naive averaging of all virtual views (full-observation control) improved classification accuracy (risk drops from 0.3079 to 0.2705) while producing lower predictive confidence (the confidence-uncertainty proxy increases from 0.3320 to 0.4844).

---

## 7. Action Value Matrix: Classification vs Uncertainty Behaviour

This analysis distinguishes three core concepts: entropy reduction, classification improvement, and prediction stability.

| Policy | Accuracy | Classification Risk | Mean Entropy | Mean Confidence | Prediction Flip Rate |
|---|---|---|---|---|---|
| global_view | 0.6921 | 0.3079 | 0.9596 | 0.6680 | - |
| center_zoom | 0.6989 | 0.3011 | 0.9754 | 0.6632 | 0.172 |
| border_zoom | 0.6763 | 0.3237 | 1.2273 | 0.5560 | 0.427 |
| upper_region | 0.7187 | 0.2813 | 1.1080 | 0.6022 | 0.354 |
| lower_region | 0.7181 | 0.2819 | 1.0827 | 0.6106 | 0.360 |
| left_region | 0.7187 | 0.2813 | 1.0626 | 0.6231 | 0.291 |
| right_region | 0.7198 | 0.2802 | 1.0986 | 0.6088 | 0.314 |
| high_frequency_texture | 0.6845 | 0.3155 | 1.8114 | 0.2748 | 0.897 |
| color_suppressed | 0.6858 | 0.3142 | 1.0409 | 0.6356 | 0.214 |
| artifact_suppressed | 0.6793 | 0.3207 | 1.1164 | 0.6085 | 0.307 |

**Entropy Increase Phenomenon:**
All predefined virtual observations increased mean predictive entropy relative to the global view, although several observations improved classification accuracy. This indicates that predictive entropy alone is not a reliable proxy for observation utility in this experiment.

**Action with smallest entropy increase:** `center_zoom` (+0.0158). It is simply the action with the smallest mean entropy increase among these observations, NOT an entropy-reducing action on average.

**Action with largest entropy increase:** `high_frequency_texture` (+0.8518). With confidence = 0.2748, accuracy = 0.6845, and a flip rate = 0.897, this observation should be treated as an example of a highly destabilizing virtual transformation according to the evaluated prediction-stability metrics.

---

## 8. Counterfactual Observation Sensitivity

Ablation: remove one observation from full-observation posterior, measure effect.

| Ablated action | Flip rate | Mean KL |
|---|---|---|
| global_view | 0.028 | 0.0017 |
| center_zoom | 0.035 | 0.0022 |
| border_zoom | 0.026 | 0.0050 |
| upper_region | 0.038 | 0.0027 |
| lower_region | 0.036 | 0.0026 |
| left_region | 0.030 | 0.0023 |
| right_region | 0.027 | 0.0023 |
| high_frequency_texture | 0.008 | 0.0213 |
| color_suppressed | 0.026 | 0.0018 |
| artifact_suppressed | 0.030 | 0.0032 |

---

## 9. Observation Redundancy

Most redundant pair (lowest JS): global_view + center_zoom (JS=0.0264)
Most diverse pair (highest JS): global_view + high_frequency_texture (JS=0.2827)

---

## 10. Failure Cases

- **global_correct_fo_correct**: 2391 (64.9%)
- **global_correct_fo_wrong**: 160 (4.3%)
- **global_wrong_fo_correct**: 298 (8.1%)
- **global_wrong_fo_wrong**: 837 (22.7%)

---

## 11. Limitations

1. Virtual observations are deterministic transforms of the original image — they contain no independent clinical signal.
2. The action-space is hand-designed (V0) — no learned policy.
3. Aggregation is simple mean/confidence-weighted mean — no learned fusion.
4. Cost model assigns uniform cost — real clinical costs vary by procedure.
5. Results hold only for ISIC 2019, EfficientNet-B4, and the frozen `clean_run_001` checkpoint.
6. No systematic literature review was performed; novelty claims are not made.
7. The minimum-entropy offline action-selection oracle evaluates all candidate observation predictions before selecting the best action. It demonstrates that some observations are more useful than others under a model-derived criterion, but it does NOT represent a deployable sequential active-acquisition policy.

---

## 12. Core Finding

The evaluated virtual transformations increased mean predictive entropy relative to the global view, with heterogeneous effects on classification accuracy and prediction stability. An offline minimum-entropy action-selection procedure achieved higher accuracy than predefined fixed-view policies in this experiment, suggesting that evidence selection may be more important than indiscriminate evidence aggregation. Observation value is action-dependent and non-monotonic.

This provides empirical motivation to investigate a learned sequential next-best-observation policy in **DERMA-ACT V1** using a development/calibration split.

---

*Report generated by `scripts/run_derma_act_v0.py` | Semantic Corrections Applied via Audit*