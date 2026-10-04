from __future__ import annotations

from threading import Lock

from app.config import settings
from app.speaker_diarization.model import SpeakerDiarizationError, SpeakerDiarizationModel

_model: SpeakerDiarizationModel | None = None
_model_lock = Lock()


def get_model() -> SpeakerDiarizationModel:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            if not settings.SPEAKER_DIARIZATION_ENABLED:
                raise SpeakerDiarizationError("Speaker-aware analysis is disabled by configuration.")
            _model = SpeakerDiarizationModel.load(
                settings.SPEAKER_DIARIZATION_MODEL,
                token=settings.PYANNOTE_TOKEN,
                device=settings.SPEAKER_DIARIZATION_DEVICE,
                sample_rate=settings.SPEAKER_DIARIZATION_SAMPLE_RATE,
                min_speakers=settings.SPEAKER_DIARIZATION_MIN_SPEAKERS,
                max_speakers=settings.SPEAKER_DIARIZATION_MAX_SPEAKERS,
            )
    return _model


def clear_model_cache() -> None:
    global _model
    with _model_lock:
        _model = None


def get_status() -> dict:
    if not settings.SPEAKER_DIARIZATION_ENABLED:
        return {
            "available": False,
            "enabled": False,
            "message": "Speaker-aware analysis is disabled by configuration.",
        }
    try:
        model = get_model()
    except SpeakerDiarizationError as exc:
        return {
            "available": False,
            "enabled": True,
            "model_id": settings.SPEAKER_DIARIZATION_MODEL,
            "message": str(exc),
        }
    return {
        "available": True,
        "enabled": True,
        "metadata": model.metadata(),
    }
