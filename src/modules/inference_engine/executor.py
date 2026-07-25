"""
Main execution engine for inference pipelines.
"""
from __future__ import annotations

import concurrent.futures
import logging
import time
from collections.abc import Sequence
from typing import Any

from src.framework.common.results import PipelineResult
from src.framework.metrics.execution import ExecutionMetadata, ResourceMetrics, TimingMetrics
from src.modules.inference_engine.context import PipelineConfig, PipelineContext
from src.modules.inference_engine.device_manager import DeviceManager
from src.modules.inference_engine.events import EventDispatcher, EventPayload, EventType
from src.modules.inference_engine.graph import ExecutionGraph
from src.modules.inference_engine.lifecycle import PipelineLifecycle
from src.modules.inference_engine.runtime_state import RuntimeState, RuntimeStateManager
from src.modules.inference_engine.stages import PipelineStage
from src.modules.inference_engine.validator import RuntimeValidator

logger = logging.getLogger(__name__)


class PipelineExecutor:
    """
    Runtime executor for AI inference pipelines.
    Handles sequential execution, timeout, recovery, and telemetry.
    """

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        dispatcher: EventDispatcher | None = None
    ) -> None:
        self._dispatcher = dispatcher or EventDispatcher()
        self._lifecycle = PipelineLifecycle()

        # Validation
        RuntimeValidator.validate_stages(stages)
        self._graph = ExecutionGraph(stages)

        self._state_manager = RuntimeStateManager()
        self._state_manager.transition_to(RuntimeState.INITIALIZING)

        for stage in self._graph.stages:
            stage.initialize()

        self._state_manager.transition_to(RuntimeState.READY)

    @property
    def lifecycle(self) -> PipelineLifecycle:
        return self._lifecycle

    @property
    def state(self) -> RuntimeState:
        return self._state_manager.current

    def execute(self, raw_image: Any, config: PipelineConfig) -> PipelineResult:
        """Execute the pipeline sequentially and produce a unified result."""
        RuntimeValidator.validate_config(config)

        context = PipelineContext(raw_image=raw_image, config=config)
        context.device_manager = DeviceManager(config.device_str)

        self._state_manager.transition_to(RuntimeState.RUNNING)

        stage_timings: dict[str, float] = {}
        skipped: list[str] = []
        errors: list[str] = []

        try:
            self._run_lifecycle_hooks(self._lifecycle.before_pipeline, context)
            self._dispatcher.emit(EventPayload(
                EventType.BEFORE_PREDICTION, context.request_id, str(time.time()),
                extra={"execution_id": context.execution_id}
            ))

            for stage in self._graph.stages:
                if context.cancellation_token.is_cancelled:
                    self._handle_cancel(context)
                    break

                if not stage.validate(context):
                    skipped.append(stage.name)
                    self._dispatcher.emit(EventPayload(
                        EventType.STAGE_SKIPPED, context.request_id, str(time.time()),
                        stage_name=stage.name
                    ))
                    continue

                self._run_lifecycle_hooks(self._lifecycle.before_stage, stage, context)
                self._dispatcher.emit(EventPayload(
                    EventType.BEFORE_STAGE, context.request_id, str(time.time()),
                    stage_name=stage.name
                ))

                start_time = time.time()
                try:
                    context = self._execute_stage_with_retry(stage, context)
                except Exception as e:
                    errors.append(stage.name)
                    logger.error(f"Stage {stage.name} failed: {e}")
                    context.add_error(f"{stage.name}: {e!s}")
                    self._dispatcher.emit(EventPayload(
                        EventType.STAGE_ERROR, context.request_id, str(time.time()),
                        stage_name=stage.name, error=e
                    ))

                    if stage.policy.is_critical and not stage.policy.skip_on_failure:
                        raise e  # Fail the pipeline

                duration = time.time() - start_time
                stage_timings[stage.name] = duration

                self._run_lifecycle_hooks(self._lifecycle.after_stage, stage, context)
                if stage.name not in errors:
                    self._dispatcher.emit(EventPayload(
                        EventType.AFTER_STAGE, context.request_id, str(time.time()),
                        stage_name=stage.name, extra={"duration": duration}
                    ))

            if self.state == RuntimeState.RUNNING:
                self._state_manager.transition_to(RuntimeState.COMPLETED)

        except Exception as e:
            if self.state != RuntimeState.CANCELLED:
                self._state_manager.transition_to(RuntimeState.FAILED)
            self._run_lifecycle_hooks(self._lifecycle.on_failure, context, e)
            self._dispatcher.emit(EventPayload(
                EventType.PIPELINE_ERROR, context.request_id, str(time.time()), error=e
            ))
            context.add_error(f"Pipeline failed: {e!s}")

        finally:
            self._run_lifecycle_hooks(self._lifecycle.before_cleanup, context)
            for stage in self._graph.stages:
                stage.cleanup()
            self._run_lifecycle_hooks(self._lifecycle.after_cleanup, context)
            self._run_lifecycle_hooks(self._lifecycle.after_pipeline, context)
            self._dispatcher.emit(EventPayload(
                EventType.AFTER_PREDICTION, context.request_id, str(time.time())
            ))

        return self._build_result(context, stage_timings, skipped, errors)

    def _execute_stage_with_retry(self, stage: PipelineStage, context: PipelineContext) -> PipelineContext:
        """Executes a stage with timeout and retry support."""
        attempts = 0
        max_attempts = stage.policy.retry_count + 1

        while attempts < max_attempts:
            try:
                attempts += 1
                return self._execute_stage_with_timeout(stage, context)
            except Exception as e:
                if attempts >= max_attempts:
                    raise e
                logger.warning(f"Retrying stage {stage.name} (attempt {attempts}/{max_attempts}) due to {e}")

        return context

    def _execute_stage_with_timeout(self, stage: PipelineStage, context: PipelineContext) -> PipelineContext:
        """Executes a stage with a hard timeout using a thread pool."""
        if stage.policy.timeout_seconds <= 0:
            return stage.execute(context)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(stage.execute, context)
            try:
                return future.result(timeout=stage.policy.timeout_seconds)
            except concurrent.futures.TimeoutError as err:
                raise TimeoutError(f"Stage '{stage.name}' exceeded timeout of {stage.policy.timeout_seconds}s") from err

    def _handle_cancel(self, context: PipelineContext) -> None:
        """Handle a cancellation token signal."""
        logger.info(f"Execution {context.execution_id} cancelled.")
        self._state_manager.transition_to(RuntimeState.CANCELLED)
        context.add_warning("Pipeline execution cancelled via token.")
        self._run_lifecycle_hooks(self._lifecycle.on_cancel, context)

    def _run_lifecycle_hooks(self, hooks: list, *args: Any) -> None:
        """Run a list of lifecycle hooks safely."""
        for hook in hooks:
            try:
                hook(*args)
            except Exception as e:
                logger.warning(f"Lifecycle hook failed: {e}")

    def _build_result(
        self, context: PipelineContext, timings: dict[str, float],
        skipped: list[str], errors: list[str]
    ) -> PipelineResult:
        """Construct the unified PipelineResult from Context."""
        tm = TimingMetrics(
            stage_timings=timings,
            total_latency_seconds=sum(timings.values()),
            stage_count=len(self._graph.stages),
            skipped_stages=skipped,
            error_stages=errors
        )
        rm = ResourceMetrics.capture_current(context.config.device_str)
        meta = ExecutionMetadata(
            request_id=context.request_id,
            model_version=context.config.model_version,
            pipeline_version=context.config.pipeline_version,
            device_str=context.config.device_str,
            class_labels=context.config.class_labels,
            timestamp_utc=str(context.created_at_utc),
            timing=tm,
            resources=rm
        )

        # Safely extract from ClassificationResult and others if present,
        # but PipelineContext holds the domain typed objects directly.
        c_res = context.classification

        return PipelineResult(
            request_id=context.request_id,
            model_version=context.config.model_version,
            pipeline_version=context.config.pipeline_version,
            predicted_class=c_res.class_labels[c_res.predicted_class] if c_res else "unknown",
            predicted_class_index=c_res.predicted_class if c_res else -1,
            class_probabilities=dict(zip(c_res.class_labels, c_res.probabilities, strict=False)) if c_res else {},
            raw_confidence=c_res.confidence if c_res else 0.0,
            calibrated_confidence=context.calibrated_confidence if context.calibrated_confidence else 0.0,
            quality=context.quality,
            ood=context.ood,
            uncertainty=context.uncertainty,
            explanation=context.explanation,
            recommendation=context.recommendation,
            execution=meta,
            warnings=context.warnings,
            errors=context.errors
        )
