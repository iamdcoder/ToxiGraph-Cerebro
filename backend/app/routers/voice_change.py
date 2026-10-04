from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.longitudinal_emotion.service import LongitudinalEmotionError, LongitudinalEmotionService
from app.models.longitudinal_emotion import VoiceHistorySession
from app.models.voice_change import VoiceChangeReport, VoiceChangeRequest, VoiceChangeResponse
from app.voice_change.engine import VoiceChangeEngine

router = APIRouter()
_history_service = LongitudinalEmotionService(settings.VOICE_HISTORY_DIR)
_engine = VoiceChangeEngine()


def _report_model(data) -> VoiceChangeReport:
    return VoiceChangeReport(
        available=data.available,
        profile_id=data.profile_id,
        current_session_id=data.current_session_id,
        current_recorded_at=data.current_recorded_at,
        historical_session_count=data.historical_session_count,
        reference_time_span_days=data.reference_time_span_days,
        level=data.level,
        change_score=data.change_score,
        confidence=data.confidence,
        current_emotion=data.current_emotion,
        historical_emotion_share=data.historical_emotion_share,
        emotion_novelty=data.emotion_novelty,
        quality_limited=data.quality_limited,
        strongest_changes=[item.to_dict() for item in data.strongest_changes],
        summary=data.summary,
        caveat=data.caveat,
    )


def _history_sessions(profile_id: str):
    try:
        sessions, _ = _history_service.get(profile_id)
        return sessions
    except LongitudinalEmotionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/status")
def change_status() -> dict:
    return {
        "available": True,
        "service": "voice_change_detection",
        "version": "voice-change-v1",
        "minimum_prior_sessions": 3,
        "message": "Detects unusual changes in recorded vocal behavior relative to prior sessions.",
    }


@router.get("/{profile_id}/change", response_model=VoiceChangeResponse)
def analyze_latest_saved_change(profile_id: str) -> VoiceChangeResponse:
    sessions = _history_sessions(profile_id)
    if not sessions:
        raise HTTPException(status_code=404, detail="No voice-history session is available for this profile.")
    current = sessions[-1]
    data = _engine.analyze(profile_id, current, sessions[:-1])
    return VoiceChangeResponse(
        success=True,
        message="Latest saved session compared with prior voice-history sessions.",
        report=_report_model(data),
    )


@router.post("/{profile_id}/change", response_model=VoiceChangeResponse)
def analyze_current_change(profile_id: str, payload: VoiceChangeRequest) -> VoiceChangeResponse:
    now = payload.recorded_at or datetime.now(timezone.utc)
    session = VoiceHistorySession(
        session_id=payload.session_id or "current-session",
        recorded_at=now,
        duration_seconds=payload.duration_seconds,
        emotion=payload.emotion,
        confidence=payload.confidence,
        acoustic=payload.acoustic,
        affect=payload.affect,
        quality_score=payload.quality_score,
        baseline_deviation=payload.baseline_deviation,
    )
    sessions = _history_sessions(profile_id)
    data = _engine.analyze(profile_id, session, sessions)
    return VoiceChangeResponse(
        success=True,
        message="Current recording compared with prior voice-history sessions.",
        report=_report_model(data),
    )
