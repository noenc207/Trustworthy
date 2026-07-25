"""Experiment registry for tracking experiments."""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Experiment:
    """Represents a single experiment."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    random_seed: int | None = None
    git_commit: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    parents: list[str] = field(default_factory=list)

    @property
    def config_hash(self) -> str:
        """Calculates the hash of the experiment configuration.

        Returns:
            str: SHA256 hash of the configuration dictionary.
        """
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()


class ExperimentRegistry:
    """Registry to track all experiments."""

    def __init__(self) -> None:
        """Initializes the experiment registry."""
        self.experiments: dict[str, Experiment] = {}

    def register(self, experiment: Experiment) -> None:
        """Registers a new experiment.

        Args:
            experiment (Experiment): The experiment to register.
        """
        self.experiments[experiment.id] = experiment

    def get(self, experiment_id: str) -> Experiment | None:
        """Retrieves an experiment by ID.

        Args:
            experiment_id (str): The ID of the experiment to retrieve.

        Returns:
            Optional[Experiment]: The experiment if found, else None.
        """
        return self.experiments.get(experiment_id)

    def get_all(self) -> list[Experiment]:
        """Retrieves all registered experiments.

        Returns:
            List[Experiment]: A list of all experiments.
        """
        return list(self.experiments.values())
