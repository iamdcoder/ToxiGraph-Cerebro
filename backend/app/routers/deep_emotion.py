from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.deep_emotion.model import DeepEmotionModelError
from app.deep_emotion.service import get_model
from app.models.features import DeepEmotionResponse, EmotionProbability


router = APIRouter()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",
}

MODEL_SOURCE = "https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition"
MODEL_NOTE = (
    "Fine-tuned Wav2Vec2 speech-emotion model. Its model card reports 79.94% validation accuracy; "
    "that is a source-reported figure, not a CEREBRO evaluation result."
)


@router.get("/status")
def deep_emotion_status() -> dict:
    try:
        model = get_model()
    except DeepEmotionModelError as exc:
        return {
            "available": False,
            "model_type": "deep_speech",
            "model_name": "Dpngtm/wav2vec2-emotion-recognition",
            "message": str(exc),
        }

    return {
        "available": True,
        "model_type": "deep_speech",
        "metadata": model.metadata(),
        "model_source": MODEL_SOURCE,
        "model_note": MODEL_NOTE,
    }


@router.post("", response_model=DeepEmotionResponse)
async def classify_deep_emotion(
    file: UploadFile = File(...),
) -> DeepEmotionResponse:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio type. Upload a PCM WAV recording.")

    try:
        try:
            model = get_model()
        except DeepEmotionModelError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise AudioProcessingError("Audio file is too large. Maximum size is 12 MB.")

        processed = AudioProcessor().process(audio_bytes)
        if processed.is_silent:
            raise HTTPException(
                status_code=422,
                detail="Audio is nearly silent. Record a clear voice sample before emotion inference.",
            )

        prediction = model.predict(processed.samples, processed.sample_rate)
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except DeepEmotionModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    ordered = sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)
    return DeepEmotionResponse(
        success=True,
        message="Deep speech emotion inference completed.",
        emotion=prediction.emotion,
        confidence=prediction.confidence,
        probabilities=[EmotionProbability(emotion=label, probability=value) for label, value in ordered],
        model_name=prediction.metadata.model_id,
        model_device=prediction.metadata.device,
        sample_rate=prediction.metadata.sample_rate,
        chunks_analyzed=prediction.chunks_analyzed,
        labels=list(prediction.metadata.labels),
        model_source=MODEL_SOURCE,
        model_note=MODEL_NOTE,
    )
