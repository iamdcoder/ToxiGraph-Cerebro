from __future__ import annotations

from threading import Lock

from app.config import settings
from app.deep_emotion.model import DeepEmotionModelError, DeepSpeechEmotionModel

_model: DeepSpeechEmotionModel | None = None
_model_lock = Lock()


def get_model() -> DeepSpeechEmotionModel:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            _model = DeepSpeechEmotionModel.load(
                settings.DEEP_EMOTION_MODEL,
                device=settings.DEEP_EMOTION_DEVICE,
                sample_rate=settings.DEEP_EMOTION_SAMPLE_RATE,
                max_seconds=settings.DEEP_EMOTION_MAX_CHUNK_SECONDS,
                chunk_seconds=settings.DEEP_EMOTION_CHUNK_SECONDS,
                overlap_seconds=settings.DEEP_EMOTION_CHUNK_OVERLAP_SECONDS,
            )
    return _model


def clear_model_cache() -> None:
    global _model
    with _model_lock:
        _model = None
