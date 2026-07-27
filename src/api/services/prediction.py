"""
Prediction Service.
Coordinates image loading, preprocessing, model inference, and database persistence.
"""
from __future__ import annotations

from pathlib import Path
from uuid import UUID

import cv2
import numpy as np
import torch
from anyio import to_thread

from src.api.db.models.prediction import Prediction
from src.api.db.repositories.prediction import PredictionCreate, PredictionRepository
from src.api.db.repositories.upload import UploadRepository
from src.core.config import get_settings
from src.core.constants import DEFAULT_IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD
from src.core.exceptions import DatasetNotFoundError, InferenceError
from src.modules.inference_engine.engine import TrustworthyInferenceEngine

_settings = get_settings()


def _load_and_preprocess_image(file_path: Path) -> tuple[np.ndarray, torch.Tensor]:
    """
    Synchronous helper to load and preprocess the image.
    Returns:
        (image_array, input_tensor)
        image_array: raw BGR numpy array (for quality/gradcam)
        input_tensor: normalized torch tensor (1, 3, H, W)
    """
    if not file_path.exists():
        raise DatasetNotFoundError(message=f"Image file not found: {file_path}")

    # Load image using OpenCV (BGR format)
    image_array = cv2.imread(str(file_path))
    if image_array is None:
        raise InferenceError(message="Failed to decode image")

    # Resize
    img_resized = cv2.resize(image_array, (DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE))

    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Normalize manually (avoiding torchvision transforms overhead for simple infer)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_normalized = (img_normalized - np.array(IMAGE_MEAN)) / np.array(IMAGE_STD)

    # Convert to CHW tensor
    img_chw = np.transpose(img_normalized, (2, 0, 1))

    # Add batch dimension
    input_tensor = torch.from_numpy(img_chw).unsqueeze(0).float()

    return image_array, input_tensor


class PredictionService:
    """Service handling prediction workflows."""

    def __init__(
        self,
        prediction_repo: PredictionRepository,
        upload_repo: UploadRepository,
        engine: TrustworthyInferenceEngine,
    ) -> None:
        self.prediction_repo = prediction_repo
        self.upload_repo = upload_repo
        self.engine = engine
        self.upload_dir = _settings.storage.upload_dir

    async def run_prediction(self, upload_id: UUID | str, user_id: UUID | str) -> Prediction:
        """
        Runs the full inference pipeline for an uploaded image and stores the result.
        """
        # Fetch upload record
        upload_record = await self.upload_repo.get(upload_id)
        if not upload_record:
            raise DatasetNotFoundError(message="Upload not found")

        if str(upload_record.user_id) != str(user_id):
            # Do not allow users to run predictions on other users' uploads
            raise DatasetNotFoundError(message="Upload not found")

        # Check if prediction already exists
        existing_prediction = await self.prediction_repo.get_by_upload(upload_id)
        if existing_prediction:
            return existing_prediction

        file_path = self.upload_dir / upload_record.filename

        # Load and preprocess image in a thread pool
        image_array, input_tensor = await to_thread.run_sync(
            _load_and_preprocess_image, file_path
        )

        # Run inference in a thread pool (since engine is mostly sync PyTorch)
        # We pass generate_explanation=False by default to speed up standard inference
        try:
            result = await to_thread.run_sync(
                self.engine.predict,
                image_array,
                input_tensor,
                True, # generate_explanation
            )
        except Exception as e:
            # Wrap any lower-level errors if not already wrapped
            import logging
            logging.exception("Inference engine crashed")
            if hasattr(e, 'status_code'):
                raise e
            raise InferenceError(message="Inference engine failure", detail=str(e)) from e

        # Prepare full result JSON payload
        full_result = {
            "probabilities": result.probabilities,
            "raw_confidence": result.raw_confidence,
            "quality": {
                "score": result.quality_report.score,
                "is_acceptable": result.quality_report.is_acceptable,
                "issues": result.quality_report.issues,
            },
            "ood": {
                "score": result.ood_result.ood_score,
                "is_ood": result.ood_result.is_ood,
                "method": result.ood_result.method_name,
            },
            "uncertainty": {
                "predictive_entropy": result.uncertainty.predictive_entropy,
                "mutual_information": result.uncertainty.mutual_information,
                "epistemic_variance": result.uncertainty.epistemic_variance.tolist() if isinstance(result.uncertainty.epistemic_variance, np.ndarray) else result.uncertainty.epistemic_variance,
                "aleatoric_variance": result.uncertainty.aleatoric_variance.tolist() if isinstance(result.uncertainty.aleatoric_variance, np.ndarray) else result.uncertainty.aleatoric_variance,
            },
            "recommendation": {
                "urgency": result.recommendation.urgency_level.value,
                "action": result.recommendation.recommended_action,
                "follow_up_days": result.recommendation.follow_up_days,
                "disclaimer": result.recommendation.disclaimer,
            }
        }

        # Create Prediction record
        prediction_in = PredictionCreate(
            user_id=upload_record.user_id,
            upload_id=upload_record.id,
            predicted_class=result.predicted_class,
            calibrated_confidence=result.calibrated_confidence,
            is_ood=result.ood_result.is_ood,
            full_result=full_result,
        )

        return await self.prediction_repo.create(obj_in=prediction_in)
