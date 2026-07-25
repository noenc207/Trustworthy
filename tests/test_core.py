"""Unit tests for the core module.

This module verifies the behavior of the core configuration, exceptions,
interfaces, and data transfer objects.
"""
from typing import Any

import pytest

from src.core.config import AppConfig, get_config
from src.core.dtos import PredictionRequestDTO, PredictionResponseDTO
from src.core.exceptions import (
    ConfigError,
    ExplainabilityError,
    ModelError,
    StorageError,
    TrustworthyError,
    ValidationError,
)
from src.core.interfaces import (
    ExplainabilityInterface,
    ModelInterface,
    StorageInterface,
)


def test_app_config_defaults() -> None:
    """Test that AppConfig provides expected defaults."""
    config = get_config()
    assert config.app_name == "Trustworthy Skin Cancer Detection"
    assert config.debug is False
    assert config.database_url == "sqlite:///./trustworthy.db"
    assert config.secret_key == "super-secret-key-for-dev-only"
    assert config.model_path == "./models/default_model.pt"


def test_app_config_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that AppConfig respects environment variables."""
    monkeypatch.setenv("APP_NAME", "Test App")
    monkeypatch.setenv("DEBUG", "true")

    config = AppConfig()
    assert config.app_name == "Test App"
    assert config.debug is True


def test_trustworthy_exception() -> None:
    """Test the base exception."""
    exc = TrustworthyError("Base error")
    assert str(exc) == "Base error"
    assert exc.message == "Base error"


def test_custom_exceptions() -> None:
    """Test custom exception types."""
    exceptions = [
        ConfigError,
        ModelError,
        StorageError,
        ExplainabilityError,
        ValidationError,
    ]
    for exc_class in exceptions:
        exc = exc_class("An error occurred")
        assert isinstance(exc, TrustworthyError)
        assert str(exc) == "An error occurred"


def test_prediction_request_dto() -> None:
    """Test the PredictionRequestDTO."""
    dto = PredictionRequestDTO(
        image_data=b"fakeimage",
        patient_id="12345",
        metadata={"age": 45}
    )
    assert dto.image_data == b"fakeimage"
    assert dto.patient_id == "12345"
    assert dto.metadata == {"age": 45}


def test_prediction_request_dto_defaults() -> None:
    """Test the PredictionRequestDTO defaults."""
    dto = PredictionRequestDTO(image_data=b"fakeimage")
    assert dto.image_data == b"fakeimage"
    assert dto.patient_id == ""
    assert dto.metadata is None


def test_prediction_response_dto() -> None:
    """Test the PredictionResponseDTO."""
    dto = PredictionResponseDTO(
        prediction_id="pred-1",
        label="benign",
        confidence=0.95,
        explanation_map=b"heatmap"
    )
    assert dto.prediction_id == "pred-1"
    assert dto.label == "benign"
    assert dto.confidence == 0.95
    assert dto.explanation_map == b"heatmap"


def test_prediction_response_dto_defaults() -> None:
    """Test the PredictionResponseDTO defaults."""
    dto = PredictionResponseDTO(
        prediction_id="pred-2",
        label="malignant",
        confidence=0.8
    )
    assert dto.prediction_id == "pred-2"
    assert dto.label == "malignant"
    assert dto.confidence == 0.8
    assert dto.explanation_map is None


class DummyStorage:
    """Dummy storage for testing StorageInterface."""
    def save(self, key: str, data: bytes) -> None:
        """Dummy save."""
        pass

    def load(self, key: str) -> bytes:
        """Dummy load."""
        return b"data"


class DummyModel:
    """Dummy model for testing ModelInterface."""
    def predict(self, input_data: Any) -> dict[str, Any]:
        """Dummy predict."""
        return {"result": True}

    def load_model(self, model_path: str) -> None:
        """Dummy load model."""
        pass


class DummyExplainability:
    """Dummy explainability engine for testing ExplainabilityInterface."""
    def generate_explanation(self, input_data: Any, prediction: dict[str, Any]) -> dict[str, Any]:
        """Dummy generate."""
        return {"heatmap": b"fake"}


def test_storage_interface_compliance() -> None:
    """Test that a dummy class complies with StorageInterface."""
    impl = DummyStorage()
    assert isinstance(impl, StorageInterface)


def test_model_interface_compliance() -> None:
    """Test that a dummy class complies with ModelInterface."""
    impl = DummyModel()
    assert isinstance(impl, ModelInterface)


def test_explainability_interface_compliance() -> None:
    """Test that a dummy class complies with ExplainabilityInterface."""
    impl = DummyExplainability()
    assert isinstance(impl, ExplainabilityInterface)
