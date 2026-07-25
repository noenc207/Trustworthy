"""Core protocols and abstract base classes.

This module defines the interfaces for key components like Models, Storage,
and Explainability engines to ensure loose coupling.
"""
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class StorageInterface(Protocol):
    """Protocol defining the interface for storage operations.

    Implementations of this protocol are responsible for persisting
    and retrieving data securely.
    """

    def save(self, key: str, data: bytes) -> None:
        """Saves data to storage.

        Args:
            key: The unique identifier for the data.
            data: The bytes data to store.

        Raises:
            StorageError: If the save operation fails.
        """
        ...

    def load(self, key: str) -> bytes:
        """Loads data from storage.

        Args:
            key: The unique identifier for the data to load.

        Returns:
            The retrieved bytes data.

        Raises:
            StorageError: If the load operation fails or the key is not found.
        """
        ...


@runtime_checkable
class ModelInterface(Protocol):
    """Protocol defining the interface for machine learning models.

    Implementations of this protocol are responsible for generating
    predictions from input data.
    """

    def predict(self, input_data: Any) -> dict[str, Any]:
        """Runs inference on the provided input data.

        Args:
            input_data: The data to run inference on.

        Returns:
            A dictionary containing prediction results and probabilities.

        Raises:
            ModelError: If the prediction operation fails.
        """
        ...

    def load_model(self, model_path: str) -> None:
        """Loads the model weights from the specified path.

        Args:
            model_path: The filesystem path or URI to the model file.

        Raises:
            ModelError: If the model fails to load.
        """
        ...


@runtime_checkable
class ExplainabilityInterface(Protocol):
    """Protocol defining the interface for Explainability engines.

    Implementations provide human-understandable explanations for
    model predictions, such as saliency maps or feature importance.
    """

    def generate_explanation(self, input_data: Any, prediction: dict[str, Any]) -> dict[str, Any]:
        """Generates an explanation for a given prediction.

        Args:
            input_data: The original input data.
            prediction: The prediction output from the model.

        Returns:
            A dictionary containing explanation artifacts (e.g., saliency map).

        Raises:
            ExplainabilityError: If the explanation generation fails.
        """
        ...
