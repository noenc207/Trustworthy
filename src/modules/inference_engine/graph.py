"""
Immutable execution graph for pipeline stages.
"""
from __future__ import annotations

from collections.abc import Sequence

from src.modules.inference_engine.stages import PipelineStage


class ExecutionGraph:
    """
    Manages stage ordering and topological verification.
    """

    def __init__(self, stages: Sequence[PipelineStage]) -> None:
        self._stages = tuple(stages)  # immutable sequence
        self._verify_topology()

    @property
    def stages(self) -> tuple[PipelineStage, ...]:
        return self._stages

    def _verify_topology(self) -> None:
        """
        Validate dependencies and detect cycles for a sequential pipeline.
        Since execution is sequential in the provided order, any dependency
        must appear before the stage that depends on it.
        """
        stage_names = {s.name for s in self._stages}

        seen = set()
        for stage in self._stages:
            for dep in stage.dependencies:
                if dep not in stage_names:
                    raise ValueError(f"Stage '{stage.name}' has missing dependency '{dep}'")
                if dep not in seen:
                    raise ValueError(
                        f"Topological error: Stage '{stage.name}' depends on '{dep}', "
                        f"which is not executed before it."
                    )
            seen.add(stage.name)
