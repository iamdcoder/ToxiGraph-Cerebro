from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.longitudinal_emotion.service import LongitudinalEmotionError, LongitudinalEmotionService, MAX_SESSIONS
from app.models.longitudinal_emotion import (
    VoiceHistoryResponse,
    VoiceHistorySession,
    VoiceHistorySessionCreate,
    VoiceHistorySessionResponse,
    VoiceHistoryTrends,
)

router = APIRouter()
_service = LongitudinalEmotionService(settings.VOICE_HISTORY_DIR)


@router.get("/status")
def history_status() -> dict:
    return {
        "available": True,
        "service": "longitudinal_voice_intelligence",
        "version": "voice-history-v1",
        "max_sessions": MAX_SESSIONS,
        "stores_audio": False,
        "message": "CEREBRO stores compact analysis summaries only; raw audio is not persisted by this service.",
    }


def _response(profile_id: str, sessions, trends) -> VoiceHistoryResponse:
    return VoiceHistoryResponse(
        available=bool(sessions),
        profile_id=profile_id,
        session_count=len(sessions),
        max_sessions=MAX_SESSIONS,
        sessions=[VoiceHistorySession(**item.to_dict()) for item in sessions],
        trends=VoiceHistoryTrends(**trends),
        message=(
            "Voice history is active." if sessions else "No voice-history sessions have been saved for this profile."
        ),
    )


@router.get("/{profile_id}", response_model=VoiceHistoryResponse)
def get_history(profile_id: str) -> VoiceHistoryResponse:
    try:
        sessions, trends = _service.get(profile_id)
        return _response(profile_id, sessions, trends)
    except LongitudinalEmotionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{profile_id}/sessions", response_model=VoiceHistorySessionResponse)
def save_session(profile_id: str, payload: VoiceHistorySessionCreate) -> VoiceHistorySessionResponse:
    try:
        session, trends = _service.append(profile_id, payload.model_dump(mode="json"))
    except LongitudinalEmotionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return VoiceHistorySessionResponse(
        success=True,
        message="Analysis summary saved to longitudinal voice history.",
        session=VoiceHistorySession(**session.to_dict()),
        trends=VoiceHistoryTrends(**trends),
    )


@router.delete("/{profile_id}")
def delete_history(profile_id: str) -> dict:
    try:
        _service.delete(profile_id)
    except LongitudinalEmotionError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"success": True, "message": "Longitudinal voice history deleted."}
