from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.config import settings
from app.dimensional_emotion.model import DimensionalEmotionError
from app.dimensional_emotion.service import get_model, get_status
from app.dimensional_emotion.interpretation import interpret_affect
from app.models.affect import (
    DimensionalAffectInterpretation,
    DimensionalAffectResponse,
    DimensionalAffectValues,
)


router = APIRouter()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",
}


def _metadata_from_audio(processed):
    from app.models.audio import AudioMetadata

    return AudioMetadata(
        format="wav",
        encoding="pcm",
        sample_rate=processed.sample_rate,
        channels=processed.channels,
        original_sample_rate=processed.original_sample_rate,
        original_channels=processed.original_channels,
        original_sample_width_bytes=processed.original_sample_width_bytes,
        duration_seconds=processed.duration_seconds,
        rms=processed.rms,
        peak=processed.peak,
        is_silent=processed.is_silent,
    )


@router.get("/status")
def dimensional_affect_status() -> dict:
    return get_status()


@router.post("", response_model=DimensionalAffectResponse)
async def analyze_dimensional_affect(
    file: UploadFile = File(...),
) -> DimensionalAffectResponse:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio type. Upload a PCM WAV recording.")

    try:
        if not settings.DIMENSIONAL_EMOTION_ENABLED:
            raise HTTPException(status_code=503, detail="Dimensional affect analysis is disabled by configuration.")

        audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise AudioProcessingError("Audio file is too large. Maximum size is 12 MB.")

        processed = AudioProcessor().process(audio_bytes)
        if processed.is_silent:
            raise HTTPException(
                status_code=422,
                detail="Audio is nearly silent. Record a clear voice sample before dimensional affect inference.",
            )

        try:
            model = get_model()
        except DimensionalEmotionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        prediction = model.predict(processed.samples, processed.sample_rate)
        interpretation = interpret_affect(
            prediction.valence,
            prediction.arousal,
            prediction.dominance,
        )
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    return DimensionalAffectResponse(
        success=True,
        message="Continuous speech-affect estimation completed.",
        source="pretrained_dimensional_speech_model",
        model_name=prediction.metadata.model_id,
        values=DimensionalAffectValues(
            valence=prediction.valence,
            arousal=prediction.arousal,
            dominance=prediction.dominance,
        ),
        interpretation=DimensionalAffectInterpretation(**interpretation.__dict__),
        scale="0 = low/negative side of the dimension, 1 = high/positive side of the dimension; values are model estimates, not psychological measurements.",
        audio=_metadata_from_audio(processed),
    )
