"""
Execution metrics for pipeline observability.
Includes timing and resource tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TimingMetrics:
    """Records duration of individual stages and overall pipeline."""
    stage_timings: dict[str, float]
    total_latency_seconds: float
    stage_count: int
    skipped_stages: list[str]
    error_stages: list[str]

    @property
    def slowest_stage(self) -> tuple[str, float] | None:
        """Identify the stage that took the most time."""
        if not self.stage_timings:
            return None
        stage_name = max(self.stage_timings, key=lambda k: self.stage_timings[k])
        return stage_name, self.stage_timings[stage_name]

    @classmethod
    def from_timing_records(cls, records: list[Any]) -> TimingMetrics:
        """
        Build TimingMetrics from a list of StageTimingRecord instances.
        """
        timings = {}
        skipped = []
        errors = []
        total_time = 0.0

        for r in records:
            if r.skipped:
                skipped.append(r.stage_name)
            else:
                timings[r.stage_name] = r.duration_seconds
                total_time += r.duration_seconds
            if r.error is not None:
                errors.append(r.stage_name)

        return cls(
            stage_timings=timings,
            total_latency_seconds=total_time,
            stage_count=len(records),
            skipped_stages=skipped,
            error_stages=errors,
        )


@dataclass
class ResourceMetrics:
    """Records resource utilization during execution."""
    peak_memory_mb: float | None
    gpu_memory_mb: float | None
    device_str: str

    @classmethod
    def capture_current(cls, device_str: str) -> ResourceMetrics:
        """Capture current resource usage."""
        return cls(
            peak_memory_mb=None,
            gpu_memory_mb=None,
            device_str=device_str,
        )


@dataclass
class ExecutionMetadata:
    """Comprehensive observability metadata for a pipeline run."""
    request_id: str
    model_version: str
    pipeline_version: str
    device_str: str
    class_labels: list[str]
    timestamp_utc: str
    timing: TimingMetrics
    resources: ResourceMetrics
