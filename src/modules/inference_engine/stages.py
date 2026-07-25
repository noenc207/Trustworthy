"""
Reusable execution stage abstraction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.modules.inference_engine.context import PipelineContext


@dataclass
class StagePolicy:
    """Execution policy for a stage."""
    timeout_seconds: float = 30.0
    retry_count: int = 0
    skip_on_failure: bool = False
    is_critical: bool = True


class PipelineStage(ABC):
    """
    Abstract Base Class for a pipeline execution stage.
    Backend-agnostic, communicates strictly through PipelineContext.
    """

    def __init__(
        self,
        name: str,
        policy: StagePolicy | None = None,
        dependencies: list[str] | None = None
    ) -> None:
        self.name = name
        self.policy = policy or StagePolicy()
        self.dependencies = dependencies or []

    @abstractmethod
    def initialize(self) -> None:
        """Setup resources before execution starts."""

    @abstractmethod
    def validate(self, context: PipelineContext) -> bool:
        """Check if the stage should execute based on current context."""

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Perform the main stage logic and mutate the context."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources after execution completes or fails."""
