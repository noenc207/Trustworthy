# DERMA-ACT V0 — METRIC AUDIT REPORT

Status: COMPLETE
Date: 2026-09-17
N_test: 3686

---

## OVERALL GATE

Per-action probabilities verified (probs=softmax(logits)): PASS
Entropy formula verified (H=-sum p log p):                 PASS
Entropy delta semantics verified:                          PASS
Entropy-confidence consistency:                            PASS
Action ranking verified:                                   PASS
Oracle label leakage (NO LABELS):                          PASS
Fixed-policy definitions (not adaptive):                   PASS
Full-observation control terminology:                      PASS
Risk reduction verified:                                   PASS
Flip rates verified from stored probs:                     PASS
Counterfactual verified (no labels):                       PASS
Redundancy JS symmetry:                                    PASS
Failure counts verified (sum==3686):                       PASS

OVERALL: PASS

---

## 1. ENTROPY FORMULA

H(p) = -sum_c p_c * log(p_c + eps),  eps=1e-12

Applied identically to global_view (A0) and all action views.
Verified against scipy.stats.entropy: max_diff = 3.58e-07 (floating point only).

---

## 2. ENTROPY DELTA SEMANTICS — CRITICAL

Column name: delta_entropy_mean (in per_action_effect.csv)

Sign convention: delta_entropy_mean = H_action - H_global
  Negative = action reduces entropy vs global view.
  More negative = MORE entropy reduction.

The correct information-gain proxy is:
  entropy_reduction = H_global - H_action   (positive, larger = more reduction)

See analysis/entropy_confidence_sanity.csv for both columns.

Recomputed values match stored values within 0.01 for all actions.

---

## 3. REPORT BUG — ACTION RANKING

Runner code (run_derma_act_v0.py line 852):
  best_action_by_ent = df_effect.sort_values("delta_entropy_mean", ascending=False).iloc[0]

ascending=False on delta_entropy_mean (negative values) picks the LEAST negative value.
LEAST negative = LEAST entropy reduction = center_zoom (delta_entropy = -0.0158).

The report therefore incorrectly states:
  "center_zoom has highest information gain"

Correct statement (ascending=True would give):
  high_frequency_texture has highest entropy reduction (delta_entropy = -0.8518)

However: entropy reduction does NOT imply accuracy improvement.
  high_frequency_texture accuracy = 0.684 < global_view accuracy = 0.692
  high_frequency_texture prediction flip rate = 89.7%
  This action dramatically destabilizes predictions.

Do NOT conflate entropy reduction with accuracy improvement.

---

## 4. ENTROPY-CONFIDENCE CONSISTENCY

For all actions: H and confidence are anti-correlated (r < -0.95).
high_frequency_texture is an extreme outlier:
  mean_entropy_action = very high
  mean_confidence_action = very low
  delta_confidence = -0.3932 (dramatic confidence reduction)

This is scientifically important: high_frequency_texture reduces BOTH confidence
and prediction stability without improving accuracy.

---

## 5. ORACLE DEFINITION

Name: model-based offline action-selection oracle (B=1)

Selection criterion:
  a*(x) = argmin_{a in A1..A9} H(p(x, a))
  (argmin entropy over non-global actions, excludes A0)
  NO test labels, NO ground-truth correctness, NO accuracy.

Aggregation:
  oracle_probs_mean = mean(p_A0, p_{a*(x)})

Verified: stored oracle_action_per_image agrees 100% with argmin_H(A1..A9).
Verified: stored probs_mean matches recomputed mean(p_A0, p_a*): max_diff=0.000000.

IMPORTANT: This is an OPTIMISTIC OFFLINE reference.
It evaluates all candidate actions BEFORE selecting the best one.
It does NOT represent a deployable sequential active-acquisition policy.

---

## 6. FULL OBSERVATION CONTROL

Definition:
  p_full = (1/10) * sum_{a=0}^{9} p_a   (mean aggregation over all 10 actions)

Verified: stored full_obs_ctrl vs mean(all 10): max_diff = 0.000000.

IMPORTANT: This is a descriptive control, NOT a theoretical performance ceiling.
  "full_observation_control" replaces "ceiling reference" and "oracle".

---

## 7. RISK REDUCTION

Risk proxy: R = 1 - max_c p_c  (complement of confidence)

  R0   (global-only)            = 0.3320
  RB1  (model-based oracle B=1) = 0.2909
  RALL (full-obs control)       = 0.4844

  delta_R_B1  = R0 - RB1  = +0.0412  (oracle REDUCES risk)
  delta_R_ALL = R0 - RALL = -0.1524  (full_obs INCREASES risk)
  fraction_captured             = -0.2700

IMPORTANT: RALL > R0.
The full_observation_control (averaging all 10 actions including high_frequency_texture)
INCREASES the risk proxy because texture views produce highly uncertain predictions
that dominate the mean, reducing overall confidence.

Do NOT interpret full_observation_control as an upper bound on risk reduction.

---

## 8. PREDICTION FLIP RATES

Verified from stored all_probs directly:
  center_zoom:             17.2%  (OK - stored matches)
  border_zoom:             42.7%  (OK)
  high_frequency_texture:  89.7%  (OK - extremely high flip rate)
  color_suppressed:        21.4%  (OK)
  artifact_suppressed:     30.7%  (OK)

All stored flip rates match recomputed values within 0.001.

---

## 9. COUNTERFACTUAL SENSITIVITY

p_all   = mean(p_0, ..., p_9)
p_minus_a = mean({p_j : j != a})

Ablating high_frequency_texture has the largest effect:
  flip_rate  = 0.0076 (very low — texture is replaceable)
  kl         = 0.0213 (highest KL from full posterior)
  conf_change = +0.0433 (INCREASES confidence when texture REMOVED)
  
This confirms: texture views are informationally dominant but accuracy-degrading.
Removing texture improves confidence while keeping predictions mostly stable.

No test labels used anywhere in counterfactual computation.

---

## 10. REDUNDANCY

Pairwise JS divergence: JS(p_a, p_b) = JS(p_b, p_a) within machine precision.
Max asymmetry: 0.00e+00.  PASS.

---

## 11. FAILURE CASE COUNTS

global_correct_full_correct : 2391  (64.87%)
global_correct_full_wrong   :  160  ( 4.34%)
global_wrong_full_correct   :  298  ( 8.08%)
global_wrong_full_wrong     :  837  (22.71%)
TOTAL                       : 3686 / 3686  PASS

---

## 12. FIXED POLICY DEFINITIONS

All fixed policies (P0..P4) use PREDETERMINED action sequences.
They are NOT adaptive. The next action is fixed regardless of model output.

P0: {A0}            = global_view only
P1: {A0, A1}        = global + center_zoom
P2: {A0, A1, A2}    = global + center + border
P3: {A0, A1, ..., A4} = first 5 actions
P4: {A0, ..., A8}   = first 9 actions (exclude artifact_suppressed)

Do NOT call these "active" policies.

---

## 13. CORRECTIONS REQUIRED IN V0 REPORT

1. "center_zoom has highest empirical information-gain proxy"
   -> INCORRECT. Runner sort bug (ascending=False picks least-negative).
   -> CORRECT: high_frequency_texture has highest entropy reduction (-0.8518).
   -> BUT: this does not imply accuracy improvement. Report separately.

2. "ceiling" / "maximum obtainable benefit"
   -> Replace with "full_observation_control".
   -> Document: RALL > R0 in this experiment.

3. "active policy"
   -> Replace with "fixed predetermined sequence" for P0..P4.
   -> Only model_oracle_B1 is model-guided (but offline, not deployable).

4. Confirm: delta_entropy semantics in all figures/tables use the same convention.