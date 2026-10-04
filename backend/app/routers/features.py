from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.features.acoustic import extract_acoustic_features
from app.models.features import AcousticFeatureResponse, AcousticFeatureSet


router = APIRouter()

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",
}


def get_audio_processor() -> AudioProcessor:
    return AudioProcessor()


@router.post("", response_model=AcousticFeatureResponse)
async def extract_features(
    file: UploadFile = File(...),
) -> AcousticFeatureResponse:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio type. Upload a PCM WAV recording.",
        )

    try:
        audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise AudioProcessingError("Audio file is too large. Maximum size is 12 MB.")

        processed = get_audio_processor().process(audio_bytes)
        features = extract_acoustic_features(processed.samples, processed.sample_rate)
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    message = "Acoustic features extracted successfully."
    if processed.is_silent:
        message = "Audio is nearly silent, so acoustic features may be unreliable."

    return AcousticFeatureResponse(
        success=True,
        message=message,
        audio=processed.to_metadata(),
        features=AcousticFeatureSet.model_validate(features.to_dict()),
        next_step="Phase 3 will feed this acoustic representation into a measurable emotion baseline model.",
    )
