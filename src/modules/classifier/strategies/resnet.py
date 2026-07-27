from pathlib import Path
from typing import Any

import numpy as np

from src.modules.classifier.interfaces import BackendAdapter
from src.modules.classifier.result import ModelInfo, PredictionCandidate, PredictionResult
from src.modules.classifier.strategies.base import AbstractClassifierStrategy


class ResNetStrategy(AbstractClassifierStrategy):
    # Stub implementation similar to EfficientNet
    def __init__(self, class_names: list[str]):
        self.class_names = class_names
        self.model = None

    def load(self, weights_path: Path, backend_adapter: BackendAdapter) -> None:
        if self.model is None:
            self.model = backend_adapter.load_model(weights_path, None)

    def adapt_input(self, image: np.ndarray) -> Any:
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        return np.transpose(image, (0, 3, 1, 2)).astype(np.float32)

    def predict(self, tensor: Any, backend_adapter: BackendAdapter) -> Any:
        return backend_adapter.predict(self.model, tensor)

    def predict_batch(self, tensors: list[Any], backend_adapter: BackendAdapter) -> list[PredictionResult]:
        import torch
        if isinstance(tensors, list):
            if isinstance(tensors[0], np.ndarray):
                batch = np.concatenate(tensors, axis=0)
            elif torch.is_tensor(tensors[0]):
                batch = torch.cat(tensors, dim=0)
            else:
                batch = tensors
        else:
            batch = tensors

        raw_output = backend_adapter.predict(self.model, batch)
        if hasattr(raw_output, "detach"):
            raw_output = raw_output.detach().cpu().numpy()

        results = []
        for i in range(raw_output.shape[0]):
            res_dict = self.postprocess_output(raw_output[i:i+1])
            results.append(PredictionResult(
                predicted_class=res_dict["predicted_class"],
                predicted_index=res_dict["predicted_index"],
                confidence=res_dict["confidence"],
                probabilities=res_dict["probabilities"],
                top_k=res_dict["top_k"],
                warnings=[]
            ))
        return results

    def postprocess_output(self, raw_output: Any) -> dict:
        if hasattr(raw_output, "detach"):
            raw_output = raw_output.detach().cpu().numpy()
        logits = np.squeeze(raw_output)
        probs = self._normalize_logits(logits)
        predicted_idx = int(np.argmax(probs))

        return {
            "predicted_class": self.class_names[predicted_idx],
            "predicted_index": predicted_idx,
            "confidence": float(probs[predicted_idx]),
            "probabilities": {self.class_names[i]: float(probs[i]) for i in range(len(probs))},
            "top_k": [PredictionCandidate(self.class_names[i], int(i), float(probs[i])) for i in np.argsort(probs)[::-1][:5]]
        }

    def model_metadata(self) -> ModelInfo:
        return ModelInfo(
            model_name="ResNet50",
            model_version="v1.0",
            backend_name="torch",
            input_shape=(3, 224, 224),
            output_classes=len(self.class_names),
            checksum="pending",
            parameter_count=25000000,
            precision="fp32",
            quantized=False,
            supports_batch=True,
            supports_gradcam=True,
            supports_uncertainty=False,
            supports_fp16=True
        )
