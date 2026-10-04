from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.cross_session_patterns.engine import CrossSessionPatternEngine, CrossSessionPatternError, MIN_PATTERN_SESSIONS
from app.longitudinal_emotion.service import LongitudinalEmotionError, LongitudinalEmotionService
from app.models.cross_session_patterns import CrossSessionPatternReport, CrossSessionPatternResponse

router = APIRouter()
_history = LongitudinalEmotionService(settings.VOICE_HISTORY_DIR)
_engine = CrossSessionPatternEngine()


@router.get("/status")
def pattern_status() -> dict:
    return {
        "available": True,
        "service": "cross_session_pattern_engine",
        "version": "cross-session-patterns-v1",
        "minimum_sessions": MIN_PATTERN_SESSIONS,
        "message": "Cross-session pattern discovery is descriptive and runs from compact saved voice-history summaries.",
    }


@router.get("/{profile_id}", response_model=CrossSessionPatternResponse)
def get_patterns(profile_id: str) -> CrossSessionPatternResponse:
    try:
        sessions, _ = _history.get(profile_id)
        report = _engine.analyze(profile_id, sessions)
    except (LongitudinalEmotionError, CrossSessionPatternError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return CrossSessionPatternResponse(
        success=True,
        message="Cross-session patterns computed from saved voice history.",
        report=CrossSessionPatternReport(**report.to_dict()),
    )
