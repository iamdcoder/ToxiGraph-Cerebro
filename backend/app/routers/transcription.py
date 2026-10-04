from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.config import settings
from app.models.transcription import TranscriptionResponse, TranscriptionResult
from app.transcription.service import ASRTranscriptionError, get_transcriber

router = APIRouter()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",
}


def _validate_content_type(file: UploadFile) -> None:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio type. Upload a PCM WAV recording.")


@router.get("/status")
def transcription_status() -> dict:
    try:
        import importlib
        importlib.import_module("torch")
        importlib.import_module("transformers")
        dependencies_available = True
    except Exception:
        dependencies_available = False

    return {
        "configured": bool(settings.ASR_MODEL),
        "dependencies_available": dependencies_available,
        "model_id": settings.ASR_MODEL,
        "device": settings.ASR_DEVICE,
        "sample_rate": settings.ASR_SAMPLE_RATE,
        "language": settings.ASR_LANGUAGE,
        "message": "Local Whisper transcription is available when its model dependencies and checkpoint are installed.",
    }


@router.post("", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)) -> TranscriptionResponse:
    _validate_content_type(file)
    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=422, detail="Audio file is too large. Maximum size is 12 MB.")
        processed = AudioProcessor().process(payload)
        if processed.is_silent:
            raise HTTPException(status_code=422, detail="Audio is nearly silent. Record a clear voice sample before transcription.")
        transcriber = get_transcriber()
        result = transcriber.transcribe(processed.samples, processed.sample_rate)
        return TranscriptionResponse(
            success=True,
            message="Local speech transcription completed.",
            transcription=TranscriptionResult(**result.to_dict()),
        )
    except ASRTranscriptionError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
