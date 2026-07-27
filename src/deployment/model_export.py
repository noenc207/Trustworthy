"""
Model Export Pipeline.

Exports trained PyTorch models to:
  - ONNX (cross-platform inference)
  - TorchScript (mobile deployment)
  - Future: TensorRT engine (GPU inference optimization)

ONNX Export best practices:
  - Dynamic axes for variable batch sizes
  - Opset 17 (latest stable)
  - Operator validation
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger


class ModelExporter:
    """
    Unified model export utility.

    Usage:
        exporter = ModelExporter(model)
        exporter.to_onnx(output_path=Path('outputs/model.onnx'))
        exporter.to_torchscript(output_path=Path('outputs/model.pt'))
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, ...] = (1, 3, 224, 224),
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.input_shape = input_shape
        self.device = device
        self.dummy_input = torch.randn(*input_shape).to(device)

    def to_onnx(
        self,
        output_path: Path,
        opset_version: int = 17,
        dynamic_batch: bool = True,
    ) -> Path:
        """
        Export to ONNX format.

        Args:
            output_path: Destination .onnx file
            opset_version: ONNX opset version
            dynamic_batch: Allow dynamic batch dimension
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dynamic_axes = None
        if dynamic_batch:
            dynamic_axes = {
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            }

        logger.info(f"Exporting model to ONNX: {output_path}")
        torch.onnx.export(
            self.model,
            self.dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        # Validate exported model
        self._validate_onnx(output_path)
        logger.success(f"ONNX export complete: {output_path}")
        return output_path

    def to_torchscript(
        self,
        output_path: Path,
        method: str = "trace",
    ) -> Path:
        """
        Export to TorchScript.

        Args:
            output_path: Destination .pt file
            method: 'trace' or 'script'
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Exporting model to TorchScript ({method}): {output_path}")

        if method == "trace":
            scripted = torch.jit.trace(self.model, self.dummy_input)
        elif method == "script":
            scripted = torch.jit.script(self.model)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'trace' or 'script'.")

        scripted.save(str(output_path))
        logger.success(f"TorchScript export complete: {output_path}")
        return output_path

    def _validate_onnx(self, onnx_path: Path) -> None:
        """Validate ONNX model structure and run consistency check."""
        try:
            import numpy as np
            import onnx
            import onnxruntime as ort

            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)

            # Runtime check
            session = ort.InferenceSession(str(onnx_path))
            dummy_np = self.dummy_input.cpu().numpy()
            ort_outputs = session.run(None, {"input": dummy_np})

            # Compare with PyTorch output
            with torch.no_grad():
                torch_out = self.model(self.dummy_input).cpu().numpy()

            max_diff = np.abs(ort_outputs[0] - torch_out).max()
            logger.info(f"ONNX validation: max difference from PyTorch = {max_diff:.2e}")
            assert max_diff < 1e-4, f"ONNX output differs too much from PyTorch: {max_diff}"

        except ImportError:
            logger.warning("onnx or onnxruntime not installed. Skipping validation.")
