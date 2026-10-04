from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.personalization import (
    PersonalizationFeedbackRequest,
    PersonalizationFeedbackResponse,
    PersonalizationPredictRequest,
    PersonalizationPredictResponse,
    PersonalizedPrediction,
    PersonalizationStatus,
)
from app.personalization.service import PersonalizedCalibrationService, PersonalizationError


router = APIRouter()
_service = PersonalizedCalibrationService(settings.PERSONALIZATION_DIR)


def _status_model(data: dict) -> PersonalizationStatus:
    return PersonalizationStatus(**data)


def _prediction_model(data: dict) -> PersonalizedPrediction:
    return PersonalizedPrediction(**data)


@router.get("/status")
def personalization_status() -> dict:
    return {
        "available": True,
        "service": "speaker_specific_emotion_calibration",
        "version": "personalized-calibration-v1",
        "minimum_feedback": 8,
        "minimum_unique_labels": 3,
        "message": "Speaker-specific calibration learns only from explicit user feedback and remains separate from the global benchmark.",
    }


@router.get("/{profile_id}", response_model=PersonalizationStatus)
def get_personalization_status(profile_id: str) -> PersonalizationStatus:
    try:
        return _status_model(_service.status(profile_id))
    except PersonalizationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{profile_id}/feedback", response_model=PersonalizationFeedbackResponse)
def save_personalization_feedback(
    profile_id: str,
    payload: PersonalizationFeedbackRequest,
) -> PersonalizationFeedbackResponse:
    try:
        status = _service.record_feedback(
            profile_id,
            session_id=payload.session_id,
            label=payload.label,
            fused_probabilities=payload.fused_probabilities,
            acoustic=payload.acoustic,
            quality_score=payload.quality_score,
            recorded_at=payload.recorded_at.isoformat() if payload.recorded_at else None,
        )
    except PersonalizationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PersonalizationFeedbackResponse(
        success=True,
        message=(
            "Feedback saved. Speaker-specific calibration is ready."
            if status["model_ready"]
            else "Feedback saved. More diverse labeled sessions are needed before speaker-specific calibration activates."
        ),
        status=_status_model(status),
    )


@router.post("/{profile_id}/predict", response_model=PersonalizationPredictResponse)
def predict_personalized(
    profile_id: str,
    payload: PersonalizationPredictRequest,
) -> PersonalizationPredictResponse:
    try:
        prediction = _service.predict(
            profile_id,
            payload.fused_probabilities,
            payload.acoustic,
        )
    except PersonalizationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PersonalizationPredictResponse(
        success=True,
        message="Personalized speaker-specific calibration applied when sufficient feedback exists.",
        prediction=_prediction_model(prediction),
    )


@router.delete("/{profile_id}")
def delete_personalization(profile_id: str) -> dict:
    try:
        _service.delete(profile_id)
    except PersonalizationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"success": True, "message": "Speaker-specific personalization history deleted."}
