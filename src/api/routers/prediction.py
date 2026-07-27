"""
Prediction API Endpoints.

Provides the core skin lesion analysis endpoint.
Accepts uploaded image ID and returns full prediction with:
  - Classification probabilities
  - Calibrated confidence
  - OOD detection result
  - Uncertainty estimates
  - Clinical recommendation
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger

from src.api.dependencies.auth import get_current_user
from src.api.dependencies.services import get_prediction_service
from src.api.schemas.prediction import PredictionRequest, PredictionResponse
from src.api.services.prediction import PredictionService
from src.core.exceptions import DatasetNotFoundError, ImageQualityError, InferenceError

router = APIRouter()


@router.post(
    "/",
    response_model=PredictionResponse,
    summary="Analyze Skin Lesion",
    description="""
    Submit an uploaded image for full trustworthy AI analysis.

    Returns:
    - Multi-class classification probabilities
    - Temperature-scaled calibrated confidence
    - OOD detection (is image in-distribution?)
    - MC Dropout uncertainty estimates
    - Clinical recommendation

    **Note**: This is a decision support tool, not a medical diagnosis.
    """,
    responses={
        200: {"description": "Prediction successful"},
        400: {"description": "Invalid image or quality too low"},
        422: {"description": "Out-of-distribution input detected"},
        500: {"description": "Inference error"},
    },
)
async def predict(
    request: PredictionRequest,
    prediction_service: PredictionService = Depends(get_prediction_service),
    current_user: dict = Depends(get_current_user),
) -> PredictionResponse:
    """
    Run the full trustworthy inference pipeline on an uploaded image.
    """
    logger.info(
        f"Prediction requested by user={current_user['sub']} "
        f"for image_id={request.image_id}"
    )
    try:
        prediction = await prediction_service.run_prediction(
            upload_id=request.image_id,
            user_id=current_user['sub']
        )
        # Convert ORM model to Pydantic PredictionResponse schema via mapping
        return PredictionResponse(
            id=prediction.id,
            upload_id=prediction.upload_id,
            predicted_class=prediction.predicted_class,
            calibrated_confidence=prediction.calibrated_confidence,
            is_ood=prediction.is_ood,
            full_result=prediction.full_result,
            created_at=prediction.created_at,
        )
    except DatasetNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.message,
        ) from e
    except ImageQualityError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=e.message,
        ) from e
    except InferenceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=e.message,
        ) from e
