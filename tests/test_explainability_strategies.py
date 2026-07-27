from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch

from src.modules.explainability.strategies.base import BaseCAMStrategy
from src.modules.explainability.strategies.gradcam import GradCAMStrategy
from src.modules.explainability.strategies.gradcampp import GradCAMPPStrategy
from src.modules.explainability.strategies.hirescam import HiResCAMStrategy
from src.modules.explainability.strategies.scorecam import ScoreCAMStrategy


@pytest.fixture
def mock_hook_manager():
    hm = MagicMock()
    hm.adapter = MagicMock()
    hm.get_activations.return_value = np.ones((1, 2, 2, 2))
    hm.get_gradients.return_value = np.ones((1, 2, 2, 2))
    return hm

def test_base_cam_strategy(mock_hook_manager):
    strategy = BaseCAMStrategy()

    # In python imports might need mocking but we can test logic
    with patch('src.modules.explainability.metrics.validator.ExplanationValidator.validate_tensors') as mock_val:
        strategy.collect(Mock(), Mock(), 0, mock_hook_manager)
        mock_val.assert_called_once()

    assert strategy.model is not None
    assert strategy.input_tensor is not None
    assert strategy.target_class == 0

    norm_h = strategy.normalize(np.array([[-1, 2], [3, 4]]))
    assert np.allclose(norm_h, np.array([[0, 0.5], [0.75, 1.0]]))

    # Zero test
    norm_zero = strategy.normalize(np.array([[-1, -2], [-3, -4]]))
    assert np.allclose(norm_zero, np.zeros((2, 2)))

    with pytest.raises(NotImplementedError):
        strategy.compute(Mock(), Mock())

def test_gradcam_strategy():
    strategy = GradCAMStrategy()
    A = torch.ones((1, 2, 2, 2))
    G = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]])

    cam = strategy.compute(A, G)
    assert cam.shape == (2, 2)
    assert np.allclose(cam, np.full((2, 2), 3.0))

def test_gradcampp_strategy():
    strategy = GradCAMPPStrategy()
    A = torch.ones((1, 2, 2, 2))
    G = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]])

    cam = strategy.compute(A, G)
    assert cam.shape == (2, 2)
    assert cam[0, 0] > 0

def test_hirescam_strategy():
    strategy = HiResCAMStrategy()
    A = torch.ones((1, 2, 2, 2))
    G = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]]])

    cam = strategy.compute(A, G)
    assert cam.shape == (2, 2)
    assert np.allclose(cam, np.full((2, 2), 3.0))

def test_scorecam_strategy():
    strategy = ScoreCAMStrategy()
    A = torch.ones((1, 2, 2, 2))

    # Fallback compute
    cam = strategy.compute(A, None)
    assert cam.shape == (2, 2)

    # Real compute
    model = MagicMock()
    model.return_value = torch.tensor([[0.1, 0.9], [0.2, 0.8]])

    strategy.model = model
    strategy.input_tensor = torch.ones((1, 3, 4, 4))
    strategy.target_class = 1

    cam2 = strategy.compute(A, None)
    assert cam2.shape == (2, 2)
