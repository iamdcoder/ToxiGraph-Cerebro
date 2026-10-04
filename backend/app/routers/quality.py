from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.models.quality import AudioQualityReport, AudioQualityResponse
from app.quality.engine import AudioQualityError, analyze_audio_quality

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
def quality_status() -> dict:
    return {
        "available": True,
        "service": "audio_quality",
        "message": "Audio quality and recording-reliability analysis is available before emotion interpretation.",
    }


@router.post("", response_model=AudioQualityResponse)
async def analyze_quality(file: UploadFile = File(...)) -> AudioQualityResponse:
    _validate_content_type(file)
    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(payload) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=422, detail="Audio file is too large. Maximum size is 12 MB.")
        processed = AudioProcessor().process(payload)
        quality = analyze_audio_quality(processed.samples, processed.sample_rate)
        return AudioQualityResponse(
            success=True,
            message="Audio quality analysis completed.",
            quality=AudioQualityReport(**quality.to_dict()),
        )
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AudioQualityError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()
