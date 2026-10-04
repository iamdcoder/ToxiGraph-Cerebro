from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.config import settings
from app.emotion_baseline.model import BaselineModelError, ClassicalEmotionModel
from app.features.acoustic import extract_acoustic_features
from app.models.features import ClassicalEmotionResponse, EmotionProbability


router = APIRouter()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",
}


def get_model() -> ClassicalEmotionModel:
    return ClassicalEmotionModel.load(settings.CLASSICAL_MODEL_PATH)


@router.get("/status")
def emotion_model_status() -> dict:
    try:
        model = get_model()
    except BaselineModelError as exc:
        return {
            "available": False,
            "model_type": "classical",
            "message": str(exc),
        }
    return {
        "available": True,
        "model_type": "classical",
        "metadata": model.metadata(),
    }


@router.post("", response_model=ClassicalEmotionResponse)
async def classify_emotion(
    file: UploadFile = File(...),
) -> ClassicalEmotionResponse:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio type. Upload a PCM WAV recording.")

    try:
        try:
            model = get_model()
        except BaselineModelError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise AudioProcessingError("Audio file is too large. Maximum size is 12 MB.")

        processor = AudioProcessor()
        processed = processor.process(audio_bytes)
        if processed.is_silent:
            raise HTTPException(
                status_code=422,
                detail="Audio is nearly silent. Record a clear voice sample before emotion inference.",
            )

        features = extract_acoustic_features(processed.samples, processed.sample_rate)
        prediction = model.predict(features.to_vector())
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except BaselineModelError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    ordered = sorted(prediction["probabilities"].items(), key=lambda item: item[1], reverse=True)
    return ClassicalEmotionResponse(
        success=True,
        message="Classical acoustic emotion inference completed.",
        emotion=prediction["emotion"],
        confidence=prediction["confidence"],
        probabilities=[EmotionProbability(emotion=label, probability=value) for label, value in ordered],
        model_name=prediction["model_name"],
        feature_version=prediction["feature_version"],
        feature_count=len(model.bundle.feature_names),
        training_metadata=prediction["training_metadata"],
    )
