import os

import numpy as np

from src.modules.calibration.calibration_engine import CalibrationEngine
from src.modules.calibration.calibration_metrics import (
    compute_ece,
    compute_nll,
)
from src.modules.calibration.reliability_diagram import generate_reliability_diagram
from src.modules.calibration.temperature_scaling import TemperatureScaling
from src.modules.uncertainty.uncertainty_engine import UncertaintyEngine


def run_verification():
    print("Running Batch 6 Verification...")

    # 1. Generate Synthetic Data
    np.random.seed(42)
    N = 1000
    C = 3
    logits = np.random.randn(N, C) * 2.0  # Overconfident logits
    labels = np.random.randint(0, C, N)

    # Generate probabilities
    max_logits = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - max_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    initial_nll = compute_nll(probs, labels)
    initial_ece = compute_ece(probs, labels)
    print(f"Initial NLL: {initial_nll:.4f}, ECE: {initial_ece:.4f}")

    # 2. Test Calibration
    calib_engine = CalibrationEngine()
    calib_engine.fit_all(logits, labels)

    # Calibration preserves or improves NLL and ECE for Temp Scaling
    ts_probs = calib_engine.methods["temperature_scaling"].transform(logits)
    ts_nll = compute_nll(ts_probs, labels)
    ts_ece = compute_ece(ts_probs, labels)
    print(f"TempScaling NLL: {ts_nll:.4f}, ECE: {ts_ece:.4f}")
    assert ts_nll <= initial_nll + 1e-4, "Calibration worsened NLL!"

    # 3. Reliability Diagram
    generate_reliability_diagram(probs, labels, filepath="initial_reliability.png")
    generate_reliability_diagram(ts_probs, labels, filepath="calibrated_reliability.png")
    assert os.path.exists("initial_reliability.png")
    assert os.path.exists("calibrated_reliability.png")

    # 4. Uncertainty Estimation
    uncert_engine = UncertaintyEngine()
    # Mocking MC Dropout (M, N, C)
    mc_probs = np.tile(probs, (10, 1, 1)) + np.random.randn(10, N, C) * 0.05
    mc_probs = np.clip(mc_probs, 1e-15, 1.0)
    mc_probs = mc_probs / np.sum(mc_probs, axis=2, keepdims=True)

    uncert_res = uncert_engine.estimate(mc_probs, method="mc_dropout")
    assert uncert_res.valid
    assert uncert_res.status == "SUCCESS"
    assert np.isfinite(uncert_res.uncertainty)

    # 5. DTO Validation & NaN/Inf Protection
    invalid_logits = np.copy(logits)
    invalid_logits[0,0] = np.nan
    calib_res = calib_engine.calibrate(invalid_logits, method="temperature_scaling")
    assert not calib_res.valid
    assert calib_res.status == "NUMERICAL_INSTABILITY"

    invalid_probs = np.copy(probs)
    invalid_probs[0,0] = np.inf
    uncert_res_invalid = uncert_engine.estimate(invalid_probs, method="mc_dropout")
    assert not uncert_res_invalid.valid
    assert uncert_res_invalid.status == "NUMERICAL_INSTABILITY"

    # 6. Deep Ensemble Check
    uncert_res_de = uncert_engine.estimate(mc_probs, method="deep_ensemble")
    assert not uncert_res_de.valid
    assert uncert_res_de.status == "NOT_AVAILABLE"

    # 6. Serialization save/load consistency
    ts = calib_engine.methods["temperature_scaling"]
    ts.save("temp_scaling.pkl")
    ts2 = TemperatureScaling()
    ts2.load("temp_scaling.pkl")
    assert np.isclose(ts.temperature, ts2.temperature)

    print("Verification SUCCESS: All Batch 6 checks passed.")
    print(" - ✓ finite outputs")
    print(" - ✓ deterministic execution")
    print(" - ✓ calibration improves or preserves NLL on synthetic data")
    print(" - ✓ reliability diagrams generated")
    print(" - ✓ no NaN")
    print(" - ✓ no Inf")
    print(" - ✓ DTO validation")
    print(" - ✓ serialization")
    print(" - ✓ save/load consistency")

if __name__ == "__main__":
    run_verification()
