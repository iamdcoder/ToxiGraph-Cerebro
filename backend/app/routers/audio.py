from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.models.audio import AudioValidationResponse


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


@router.post("/validate", response_model=AudioValidationResponse)
async def validate_audio(
    file: UploadFile = File(...),
) -> AudioValidationResponse:
    filename = file.filename or "audio"
    content_type = (file.content_type or "").lower()

    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                "Unsupported audio type. CEREBRO Phase 1 accepts PCM WAV audio "
                "so the browser-to-backend pipeline remains dependency-light."
            ),
        )

    try:
        audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise AudioProcessingError(
                "Audio file is too large. Maximum size is 12 MB."
            )

        processed = get_audio_processor().process(audio_bytes)
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    message = "Audio validated and normalized successfully."
    if processed.is_silent:
        message = (
            "Audio is valid, but it is nearly silent. Record again with a clear voice."
        )

    return AudioValidationResponse(
        success=True,
        message=f"{message} Source file: {filename}.",
        audio=processed.to_metadata(),
        next_step=(
            "Phase 2 will extract acoustic features from this normalized waveform."
        ),
    )
