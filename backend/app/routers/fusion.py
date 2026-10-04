from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.config import settings
from app.deep_emotion.model import DeepEmotionModelError
from app.deep_emotion.service import get_model as get_deep_model
from app.emotion_baseline.model import BaselineModelError
from app.emotion_baseline.model import ClassicalEmotionModel
from app.features.acoustic import extract_acoustic_features
from app.fusion.engine import FusionError
from app.fusion.service import get_calibration_status, get_engine
from app.fusion.training import FusionTrainingError
from app.models.features import EmotionProbability, FusionEmotionResponse, FusionModelContribution

router = APIRouter()
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
    "audio/vnd.wave",
    "application/octet-stream",
}


def get_classical_model() -> ClassicalEmotionModel:
    return ClassicalEmotionModel.load(settings.CLASSICAL_MODEL_PATH)


@router.get("/status")
def fusion_status() -> dict:
    calibration = get_calibration_status()
    if not calibration["available"]:
        return {
            "available": False,
            "model_type": "hybrid_fusion",
            "message": calibration["message"],
        }
    return {
        "available": True,
        "model_type": "hybrid_fusion",
        "calibration": calibration,
    }


@router.post("", response_model=FusionEmotionResponse)
async def classify_fused_emotion(file: UploadFile = File(...)) -> FusionEmotionResponse:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio type. Upload a PCM WAV recording.")

    try:
        try:
            engine = get_engine()
        except FusionTrainingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            classical_model = get_classical_model()
        except BaselineModelError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            deep_model = get_deep_model()
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

        features = extract_acoustic_features(processed.samples, processed.sample_rate)
        classical_prediction = classical_model.predict(features.to_vector())
        deep_prediction = deep_model.predict(processed.samples, processed.sample_rate)
        fused = engine.combine(
            classical_prediction["probabilities"],
            deep_prediction.probabilities,
        )
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FusionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    ordered = sorted(fused.probabilities.items(), key=lambda item: item[1], reverse=True)
    classical_probs = sorted(classical_prediction["probabilities"].items(), key=lambda item: item[1], reverse=True)
    deep_probs = sorted(deep_prediction.probabilities.items(), key=lambda item: item[1], reverse=True)

    classical_contribution = FusionModelContribution(
        model_type="classical_acoustic",
        model_name=classical_prediction["model_name"],
        emotion=max(classical_prediction["probabilities"], key=classical_prediction["probabilities"].get),
        confidence=classical_prediction["confidence"],
        calibrated_confidence=fused.classical_confidence,
        weight=fused.classical_weight,
        temperature=fused.classical_temperature,
        probabilities=[EmotionProbability(emotion=label, probability=value) for label, value in classical_probs],
    )
    deep_contribution = FusionModelContribution(
        model_type="deep_speech",
        model_name=deep_prediction.metadata.model_id,
        emotion=deep_prediction.emotion,
        confidence=deep_prediction.confidence,
        calibrated_confidence=fused.deep_confidence,
        weight=fused.deep_weight,
        temperature=fused.deep_temperature,
        probabilities=[EmotionProbability(emotion=label, probability=value) for label, value in deep_probs],
    )

    return FusionEmotionResponse(
        success=True,
        message="Hybrid calibrated emotion fusion completed.",
        emotion=fused.emotion,
        confidence=fused.confidence,
        probabilities=[EmotionProbability(emotion=label, probability=value) for label, value in ordered],
        fusion_strategy="temperature-calibrated weighted probability fusion",
        agreement=fused.agreement,
        js_divergence=fused.js_divergence,
        classical=classical_contribution,
        deep=deep_contribution,
        calibration_version=fused.calibration_version,
        sample_rate=processed.sample_rate,
        chunks_analyzed=deep_prediction.chunks_analyzed,
    )
