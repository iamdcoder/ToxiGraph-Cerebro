from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.privacy import PrivacyDeleteResponse, PrivacyStatusResponse
from app.privacy.service import PrivacyService
from app.security import SecurityPolicyError

router = APIRouter()


def get_service() -> PrivacyService:
    return PrivacyService(
        personal_baseline_dir=settings.PERSONAL_BASELINE_DIR,
        voice_history_dir=settings.VOICE_HISTORY_DIR,
        personalization_dir=settings.PERSONALIZATION_DIR,
        retention_days=settings.PRIVACY_RETENTION_DAYS,
        retention_enabled=settings.PRIVACY_RETENTION_ENABLED,
    )


@router.get("/status")
def privacy_status() -> dict:
    return {
        "available": True,
        "service": "privacy_and_data_controls",
        "retention_enabled": settings.PRIVACY_RETENTION_ENABLED,
        "retention_days": settings.PRIVACY_RETENTION_DAYS,
        "raw_audio_stored": False,
        "raw_transcripts_stored": False,
        "http_auth_enabled": settings.API_AUTH_ENABLED,
        "websocket_auth_enabled": settings.WEBSOCKET_AUTH_ENABLED,
        "rate_limit_enabled": settings.HTTP_RATE_LIMIT_ENABLED,
        "message": "CEREBRO exposes explicit profile deletion and configurable retention for persistent personal summaries.",
    }


@router.get("/{profile_id}", response_model=PrivacyStatusResponse)
def get_privacy(profile_id: str) -> PrivacyStatusResponse:
    try:
        payload = get_service().status(profile_id)
    except SecurityPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PrivacyStatusResponse(**payload)


@router.delete("/{profile_id}", response_model=PrivacyDeleteResponse)
def delete_privacy(profile_id: str) -> PrivacyDeleteResponse:
    try:
        payload = get_service().delete(profile_id)
    except SecurityPolicyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PrivacyDeleteResponse(**payload)
