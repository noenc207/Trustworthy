
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.modules.ood_detection import (
    EnergyDetector,
    EntropyDetector,
    MahalanobisDetector,
    MSPDetector,
    ODINDetector,
)


class DummyModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x):
        return self.fc(x)

@pytest.fixture
def dummy_model():
    model = DummyModel(num_classes=3)
    model.eval()
    return model

@pytest.fixture
def dummy_data():
    x = torch.randn(4, 10)
    return x

def test_msp_detector(dummy_model, dummy_data):
    detector = MSPDetector(dummy_model)
    res = detector.detect(dummy_data)
    assert res.method == "MSP"
    assert 0.0 <= res.score <= 1.0
    assert 0.0 <= res.confidence <= 1.0

def test_energy_detector(dummy_model, dummy_data):
    detector = EnergyDetector(dummy_model, temperature=1.0)
    res = detector.detect(dummy_data)
    assert res.method == "Energy"
    assert isinstance(res.score, float)

def test_entropy_detector(dummy_model, dummy_data):
    detector = EntropyDetector(dummy_model)
    res = detector.detect(dummy_data)
    assert res.method == "Entropy"
    assert res.score >= 0.0

def test_odin_detector(dummy_model, dummy_data):
    detector = ODINDetector(dummy_model, epsilon=0.0014, temperature=1000.0)
    res = detector.detect(dummy_data)
    assert res.method == "ODIN"
    assert 0.0 <= res.score <= 1.0

def test_mahalanobis_detector(dummy_model):
    # Setup Mahalanobis with dummy layer name
    detector = MahalanobisDetector(dummy_model, feature_layer_name='fc')

    # Create Gaussian toy datasets
    # In-Distribution (Gaussian 1 and 2)
    x_train = torch.cat([torch.randn(50, 10) - 2, torch.randn(50, 10) + 2])
    y_train = torch.cat([torch.zeros(50, dtype=torch.long), torch.ones(50, dtype=torch.long)])
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=10)

    # Fit the detector (calculates covariance and inversion)
    detector.fit(loader)

    # Test In-distribution point
    x_id = torch.randn(2, 10) - 2 # should be close to class 0
    res_id = detector.detect(x_id)

    # Synthetic OOD point (far away)
    x_ood = torch.randn(2, 10) + 20
    res_ood = detector.detect(x_ood)

    assert res_id.method == "Mahalanobis"
    assert res_ood.score > res_id.score

def test_numerical_stability(dummy_model):
    # Infinitely large logits
    class NaNModel(nn.Module):
        def forward(self, x):
            return torch.full((x.size(0), 3), float('inf'))

    nan_detector = EntropyDetector(NaNModel())
    res = nan_detector.detect(torch.randn(2, 10))
    # Due to softmax(inf), we get NaNs inside, detector should fallback gracefully
    assert res.score is None
    assert res.valid is False
    assert res.status in ("NUMERICAL_INSTABILITY", "COMPUTATION_ERROR")

def test_deterministic_execution(dummy_model, dummy_data):
    torch.manual_seed(42)
    detector1 = EnergyDetector(dummy_model)
    res1 = detector1.detect(dummy_data)

    torch.manual_seed(42)
    detector2 = EnergyDetector(dummy_model)
    res2 = detector2.detect(dummy_data)

    assert res1.score == res2.score

def test_mixed_precision(dummy_model, dummy_data):
    detector = MSPDetector(dummy_model)
    if torch.cuda.is_available():
        dummy_model = dummy_model.cuda()
        dummy_data = dummy_data.cuda()
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            res = detector.detect(dummy_data)
            assert isinstance(res.score, float)
    else:
        with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
            res = detector.detect(dummy_data)
            assert isinstance(res.score, float)
