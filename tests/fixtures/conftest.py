"""
Shared pytest fixtures for all test suites.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Provide the test device (CPU for CI, GPU if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture()
def sample_image_rgb() -> np.ndarray:
    """224x224 random RGB image as uint8."""
    return np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)


@pytest.fixture()
def sample_tensor() -> torch.Tensor:
    """Batch of 2 image tensors (2, 3, 224, 224)."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture()
def sample_labels_7class() -> np.ndarray:
    """50 random labels in [0, 7) for 7-class classification."""
    return np.random.randint(0, 7, size=50)


@pytest.fixture()
def sample_probabilities_7class(sample_labels_7class) -> np.ndarray:
    """Corresponding random probability vectors summing to 1."""
    probs = np.random.dirichlet(np.ones(7), size=len(sample_labels_7class))
    return probs
