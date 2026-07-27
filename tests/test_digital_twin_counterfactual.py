import pytest
import torch
import torch.nn as nn

from src.modules.explainability.counterfactual.recourse import CounterfactualRecourseGenerator
from src.modules.explainability.digital_twin.simulator import DigitalTwinSimulator


class DummyGenerativeModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple linear transformation to act as a mock generator/VAE
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


class DummyGenerativeModelTuple(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x), torch.zeros_like(x)


class DummyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


def test_digital_twin_simulator():
    model = DummyGenerativeModel()
    simulator = DigitalTwinSimulator(generative_model=model, seed=42)

    x = torch.randn(2, 10)

    # Test simulate output
    output = simulator.simulate(x)
    assert output.shape == (2, 10)
    assert not output.requires_grad

    # Test determinism
    simulator1 = DigitalTwinSimulator(generative_model=model, seed=123)
    simulator2 = DigitalTwinSimulator(generative_model=model, seed=123)
    out1 = simulator1.simulate(x)
    out2 = simulator2.simulate(x)
    assert torch.allclose(out1, out2)

    # Test tuple return handling
    tuple_model = DummyGenerativeModelTuple()
    simulator_tuple = DigitalTwinSimulator(generative_model=tuple_model, seed=42)
    out_tuple = simulator_tuple.simulate(x)
    assert isinstance(out_tuple, torch.Tensor)
    assert out_tuple.shape == (2, 10)

    # Test empty tensor
    with pytest.raises(ValueError):
        simulator.simulate(torch.empty(0))


def test_counterfactual_recourse_generator():
    generator = DummyGenerativeModel()
    classifier = DummyClassifier()

    cf_generator = CounterfactualRecourseGenerator(
        classifier=classifier,
        generator=generator,
        lambda_reg=0.1,
        learning_rate=0.1,
        max_iter=10,
        seed=42
    )

    z_orig = torch.randn(2, 10)
    target_class = torch.tensor([1, 0])

    z_opt = cf_generator.generate(z_orig, target_class)

    assert z_opt.shape == (2, 10)
    # The output should not be exactly the same as the input, as it should be optimized
    # But it shouldn't be too far if lambda_reg is high or iterations are few
    assert not torch.allclose(z_orig, z_opt)

    # Test determinism
    cf_gen1 = CounterfactualRecourseGenerator(classifier, generator, seed=123, max_iter=5)
    cf_gen2 = CounterfactualRecourseGenerator(classifier, generator, seed=123, max_iter=5)
    z_opt1 = cf_gen1.generate(z_orig, target_class)
    z_opt2 = cf_gen2.generate(z_orig, target_class)
    assert torch.allclose(z_opt1, z_opt2)

    # Test empty tensor
    with pytest.raises(ValueError):
        cf_generator.generate(torch.empty(0), target_class)
