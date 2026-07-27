"""Unit tests for the evaluation layer."""

import numpy as np
import pytest

from src.modules.evaluation.benchmark.benchmark_loader import BenchmarkLoader
from src.modules.evaluation.benchmark.benchmark_runner import BenchmarkRunner
from src.modules.evaluation.calibration.brier_score import brier_score
from src.modules.evaluation.calibration.expected_calibration_error import expected_calibration_error
from src.modules.evaluation.calibration.maximum_calibration_error import maximum_calibration_error
from src.modules.evaluation.statistics.effect_size import cliffs_delta, cohens_d, glass_delta


def test_expected_calibration_error() -> None:
    """Test ECE computation."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    ece = expected_calibration_error(y_true, y_prob, num_bins=2)
    assert isinstance(ece, float)
    assert ece >= 0.0

def test_maximum_calibration_error() -> None:
    """Test MCE computation."""
    y_true = np.array([0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8])
    mce = maximum_calibration_error(y_true, y_prob, num_bins=2)
    assert isinstance(mce, float)
    assert mce >= 0.0

def test_brier_score() -> None:
    """Test Brier score computation."""
    y_true = np.array([0, 1])
    y_prob = np.array([0.0, 1.0])
    assert brier_score(y_true, y_prob) == 0.0

def test_cohens_d() -> None:
    """Test Cohen's d computation."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    d = cohens_d(x, y)
    assert isinstance(d, float)

def test_glass_delta() -> None:
    """Test Glass's delta computation."""
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 4, 5, 6])
    d = glass_delta(x, y)
    assert isinstance(d, float)

def test_cliffs_delta() -> None:
    """Test Cliff's delta computation."""
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    d = cliffs_delta(x, y)
    assert d == -1.0

def test_benchmark_loader() -> None:
    """Test BenchmarkLoader."""
    loader = BenchmarkLoader("ISIC")
    assert loader.load_metadata()["name"] == "ISIC"
    with pytest.raises(ValueError):
        BenchmarkLoader("UNKNOWN")

def test_benchmark_runner() -> None:
    """Test BenchmarkRunner behavior without fabricating metrics."""
    loader = BenchmarkLoader("ISIC")
    runner = BenchmarkRunner(loader)
    res = runner.run(metrics_available=False)
    assert res == "External Validation Required"
