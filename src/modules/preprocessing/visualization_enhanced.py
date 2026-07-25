"""
Enhanced Preprocessing Visualization.
Visualize before/after preprocessing, with metadata annotations and stage-level inspection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from src.modules.preprocessing.pipeline import PipelineMetadata


class PreprocessingVisualizer:
    """Advanced visualization tools for preprocessing pipelines."""

    @staticmethod
    def save_comparison(
        original_img: np.ndarray,
        processed_img: np.ndarray,
        output_path: str | Path,
        metadata: PipelineMetadata | None = None,
        stage_indices: list[int] | None = None,
    ) -> None:
        """
        Save a side-by-side comparison of original and processed images.

        Args:
            original_img: Original image (HWC, uint8 RGB).
            processed_img: Processed image (may be tensor or numpy).
            output_path: Path to save the comparison.
            metadata: Optional pipeline metadata for annotations.
            stage_indices: Optional stage indices to highlight.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert processed to numpy if it's a tensor
        if hasattr(processed_img, "numpy"):
            proc_np = processed_img.numpy()
            if proc_np.ndim == 3 and proc_np.shape[0] in [1, 3]:
                proc_np = np.transpose(proc_np, (1, 2, 0))
            processed_img = proc_np

        # Ensure uint8 range [0, 255]
        if processed_img.dtype != np.uint8:
            if processed_img.max() <= 1.0:
                processed_img = (processed_img * 255).astype(np.uint8)
            else:
                processed_img = np.clip(processed_img, 0, 255).astype(np.uint8)

        # Resize original to match processed size
        h_proc, w_proc = processed_img.shape[:2]
        original_resized = cv2.resize(original_img, (w_proc, h_proc))

        # Convert to BGR for OpenCV
        orig_bgr = cv2.cvtColor(original_resized, cv2.COLOR_RGB2BGR)
        proc_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

        # Create side-by-side comparison
        comparison = cv2.hconcat([orig_bgr, proc_bgr])

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        color = (0, 255, 0)

        cv2.putText(comparison, "Original", (20, 40), font, font_scale, color, thickness)
        cv2.putText(
            comparison,
            "Processed",
            (w_proc + 20, 40),
            font,
            font_scale,
            color,
            thickness,
        )

        # Add metadata if provided
        if metadata:
            info_text = f"Stages: {len(metadata.stages) if metadata.stages else 0} | "
            info_text += f"Time: {metadata.total_time_ms:.1f}ms"
            cv2.putText(
                comparison,
                info_text,
                (20, comparison.shape[0] - 20),
                font,
                font_scale * 0.7,
                color,
                thickness - 1,
            )

        cv2.imwrite(str(output_path), comparison)
        logger.info(f"Saved preprocessing comparison to {output_path}")

    @staticmethod
    def create_stage_breakdown_plot(
        metadata: PipelineMetadata,
        output_path: str | Path | None = None,
    ) -> None:
        """
        Create a visualization of processing time by stage.

        Args:
            metadata: Pipeline metadata with stage information.
            output_path: Optional path to save the plot.
        """
        if not metadata.stages:
            logger.warning("No stage metadata available for breakdown plot")
            return

        output_path = Path(output_path) if output_path else None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        stage_names = [s.stage_name for s in metadata.stages]
        stage_times = [s.processing_time_ms for s in metadata.stages]

        plt.figure(figsize=(12, 6))
        plt.barh(stage_names, stage_times, color="steelblue")
        plt.xlabel("Processing Time (ms)")
        plt.ylabel("Stage")
        plt.title(f"Pipeline Stage Breakdown - {metadata.pipeline_mode.upper()}")
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved stage breakdown plot to {output_path}")
        else:
            plt.show()

        plt.close()

    @staticmethod
    def save_metadata_report(
        metadata: PipelineMetadata,
        output_path: str | Path,
    ) -> None:
        """
        Save detailed metadata report as JSON and human-readable text.

        Args:
            metadata: Pipeline metadata.
            output_path: Base path for saving reports.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # JSON report
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        logger.info(f"Saved metadata JSON to {json_path}")

        # Text report
        txt_path = output_path.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Preprocessing Pipeline Report - {metadata.image_id}\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Mode: {metadata.pipeline_mode}\n")
            f.write(f"Original Shape: {metadata.original_shape}\n")
            f.write(f"Final Shape: {metadata.final_shape}\n")
            f.write(f"Total Time: {metadata.total_time_ms:.2f} ms\n")
            f.write(f"Status: {'SUCCESS' if metadata.success else 'FAILED'}\n")
            if not metadata.success:
                f.write(f"Error: {metadata.error_message}\n")
            f.write("\n" + "-" * 70 + "\n")
            f.write("Stage Breakdown:\n")
            f.write("-" * 70 + "\n\n")

            if metadata.stages:
                for i, stage in enumerate(metadata.stages):
                    f.write(f"Stage {i}: {stage.stage_name}\n")
                    f.write(f"  Time: {stage.processing_time_ms:.2f} ms\n")
                    f.write(f"  Output Shape: {stage.output_shape}\n")
                    if stage.params:
                        f.write(f"  Params: {stage.params}\n")
                    f.write(f"  Status: {'SUCCESS' if stage.success else 'FAILED'}\n")
                    if not stage.success:
                        f.write(f"  Error: {stage.error_message}\n")
                    f.write("\n")

        logger.info(f"Saved metadata report to {txt_path}")

    @staticmethod
    def create_pipeline_summary(
        pipeline_statistics: dict[str, Any],
        output_path: str | Path | None = None,
    ) -> None:
        """
        Create a summary visualization of pipeline statistics.

        Args:
            pipeline_statistics: Statistics dictionary from Pipeline.get_statistics().
            output_path: Optional path to save the plot.
        """
        output_path = Path(output_path) if output_path else None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        _fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Execution status pie chart
        statuses = [
            pipeline_statistics.get("successful_executions", 0),
            pipeline_statistics.get("failed_executions", 0),
        ]
        axes[0].pie(
            statuses,
            labels=["Successful", "Failed"],
            autopct="%1.1f%%",
            colors=["green", "red"],
        )
        axes[0].set_title("Execution Status")

        # Processing time histogram
        if pipeline_statistics.get("total_executions", 0) > 0:
            times = [
                pipeline_statistics.get("min_time_ms", 0),
                pipeline_statistics.get("avg_time_ms", 0),
                pipeline_statistics.get("max_time_ms", 0),
            ]
            axes[1].bar(["Min", "Avg", "Max"], times, color="steelblue")
            axes[1].set_ylabel("Time (ms)")
            axes[1].set_title("Processing Time Statistics")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path)
            logger.info(f"Saved pipeline summary plot to {output_path}")
        else:
            plt.show()

        plt.close()
