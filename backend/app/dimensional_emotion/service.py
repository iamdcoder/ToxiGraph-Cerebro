from __future__ import annotations

from threading import Lock

from app.config import settings
from app.dimensional_emotion.model import DimensionalEmotionError, DimensionalSpeechEmotionModel

_model: DimensionalSpeechEmotionModel | None = None
_model_lock = Lock()


def get_model() -> DimensionalSpeechEmotionModel:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            _model = DimensionalSpeechEmotionModel.load(
                settings.DIMENSIONAL_EMOTION_MODEL,
                device=settings.DIMENSIONAL_EMOTION_DEVICE,
                sample_rate=settings.DIMENSIONAL_EMOTION_SAMPLE_RATE,
                max_seconds=settings.DIMENSIONAL_EMOTION_MAX_CHUNK_SECONDS,
                chunk_seconds=settings.DIMENSIONAL_EMOTION_CHUNK_SECONDS,
                overlap_seconds=settings.DIMENSIONAL_EMOTION_CHUNK_OVERLAP_SECONDS,
            )
    return _model


def clear_model_cache() -> None:
    global _model
    with _model_lock:
        _model = None


def get_status() -> dict:
    if not settings.DIMENSIONAL_EMOTION_ENABLED:
        return {
            "available": False,
            "enabled": False,
            "message": "Dimensional affect analysis is disabled by configuration.",
        }
    try:
        model = get_model()
    except DimensionalEmotionError as exc:
        return {
            "available": False,
            "enabled": True,
            "model_id": settings.DIMENSIONAL_EMOTION_MODEL,
            "message": str(exc),
        }
    return {
        "available": True,
        "enabled": True,
        "metadata": model.metadata(),
    }
