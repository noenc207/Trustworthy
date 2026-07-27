import time

import numpy as np
import torch

from .dto import BatchPredictionResult, PredictionResult, PredictionSummary
from .exceptions import NumericalInstabilityError, PredictionError
from .label_mapper import LabelMapper


class PredictionEngine:
    def __init__(self, model: torch.nn.Module, label_mapper: LabelMapper, device: str = "cpu", mixed_precision: bool = False):
        self.model = model
        self.label_mapper = label_mapper
        self.device = device
        self.mixed_precision = mixed_precision

        self.model.to(self.device)
        self.model.eval()

    def _ensure_tensor(self, x: np.ndarray) -> torch.Tensor:
        if not np.isfinite(x).all():
            raise NumericalInstabilityError("Input contains NaN or Inf", "INVALID_INPUT")
        return torch.tensor(x, dtype=torch.float32).to(self.device)

    def predict(self, image: np.ndarray) -> PredictionResult:
        res = self.predict_batch(np.expand_dims(image, 0))
        return res.predictions[0]

    def predict_batch(self, images: np.ndarray) -> BatchPredictionResult:
        start_time = time.time()

        if len(images) == 0:
            raise PredictionError("Zero-length batch provided", "EMPTY_BATCH")

        try:
            tensor = self._ensure_tensor(images)

            with torch.no_grad(), torch.autocast(device_type="cuda" if "cuda" in self.device else "cpu", enabled=self.mixed_precision):
                outputs = self.model(tensor)
                if isinstance(outputs, tuple) and len(outputs) == 2:
                    logits, embeddings = outputs
                else:
                    logits = outputs
                    embeddings = torch.zeros((len(images), 1))

                if not torch.isfinite(logits).all():
                    raise NumericalInstabilityError("NaN or Inf encountered in logits", "INVALID_LOGITS")

                probs = torch.softmax(logits, dim=1)

                if not torch.isfinite(probs).all():
                    raise NumericalInstabilityError("NaN or Inf encountered in probabilities", "INVALID_PROBABILITIES")

                logits_np = logits.cpu().numpy()
                probs_np = probs.cpu().numpy()
                emb_np = embeddings.cpu().numpy()

            predictions = []
            for i in range(len(images)):
                pred_idx = int(np.argmax(probs_np[i]))
                pred_class = self.label_mapper.index_to_label(pred_idx)
                conf = float(probs_np[i, pred_idx])

                class_probs = {self.label_mapper.index_to_label(k): float(probs_np[i, k]) for k in range(probs_np.shape[1])}

                pred = PredictionResult(
                    prediction=pred_class,
                    class_index=pred_idx,
                    class_name=pred_class,
                    probabilities=class_probs,
                    confidence=conf,
                    logits=logits_np[i],
                    embedding=emb_np[i],
                    runtime_ms=(time.time() - start_time) * 1000.0,
                    metadata={"device": self.device, "mixed_precision": self.mixed_precision}
                )
                predictions.append(pred)

            summary = PredictionSummary(
                total_predictions=len(images),
                mean_confidence=float(np.mean([p.confidence for p in predictions])),
                latency_stats={"total_ms": (time.time() - start_time) * 1000.0}
            )

            return BatchPredictionResult(
                predictions=predictions,
                summary=summary,
                batch_runtime_ms=(time.time() - start_time) * 1000.0
            )

        except NumericalInstabilityError as e:
            # Trap and return structured failure
            failed_preds = []
            for i in range(len(images)):
                failed_preds.append(PredictionResult(
                    prediction="UNKNOWN", class_index=-1, class_name="UNKNOWN",
                    probabilities={}, confidence=0.0,
                    valid=False, status="NUMERICAL_INSTABILITY",
                    warnings=[str(e)]
                ))
            return BatchPredictionResult(
                predictions=failed_preds,
                summary=PredictionSummary(len(images), 0.0, {}),
                batch_runtime_ms=(time.time() - start_time) * 1000.0,
                valid=False, status="NUMERICAL_INSTABILITY"
            )
        except Exception as e:
            raise PredictionError(f"Prediction failed: {e!s}", "PREDICTION_FAIL")
