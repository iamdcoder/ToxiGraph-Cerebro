from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.config import settings
from app.deep_emotion.model import DeepEmotionModelError
from app.deep_emotion.service import get_model as get_deep_model
from app.emotion_baseline.model import BaselineModelError, ClassicalEmotionModel
from app.explainability.engine import ExplainabilityError, build_insights
from app.features.acoustic import extract_acoustic_features
from app.fusion.engine import FusionError
from app.fusion.service import get_engine
from app.fusion.training import FusionTrainingError
from app.models.features import EmotionProbability, FusionEmotionResponse, FusionModelContribution
from app.models.personal_baseline import PersonalBaselineComparison, PersonalBaselineMetricComparison
from app.models.personalization import PersonalizedPrediction
from app.personal_baseline.service import PersonalBaselineError, PersonalBaselineService
from app.personalization.service import PersonalizedCalibrationService, PersonalizationError
from app.quality.engine import AudioQualityError, analyze_audio_quality
from app.models.quality import AudioQualityReport
from app.models.insights import (
    AudioEmotionInsightsResponse,
    EmotionInsights,
    InsightAlternative,
    InsightEvidence,
    UncertaintyAnalysis,
    UncertaintyFactor,
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


def get_classical_model() -> ClassicalEmotionModel:
    return ClassicalEmotionModel.load(settings.CLASSICAL_MODEL_PATH)


def _metadata_from_audio(processed) -> object:
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


def _run_insights(audio_bytes: bytes, profile_id: str | None = None) -> AudioEmotionInsightsResponse:
    try:
        try:
            fusion_engine = get_engine()
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

        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise AudioProcessingError("Audio file is too large. Maximum size is 12 MB.")

        processed = AudioProcessor().process(audio_bytes)
        try:
            quality = analyze_audio_quality(processed.samples, processed.sample_rate)
        except AudioQualityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if processed.is_silent:
            raise HTTPException(
                status_code=422,
                detail="Audio is nearly silent. Record a clear voice sample before emotion inference.",
            )

        features = extract_acoustic_features(processed.samples, processed.sample_rate)
        classical_prediction = classical_model.predict(features.to_vector())
        deep_prediction = deep_model.predict(processed.samples, processed.sample_rate)
        fused = fusion_engine.combine(
            classical_prediction["probabilities"],
            deep_prediction.probabilities,
        )
        try:
            explanation = build_insights(
                fusion_prediction=fused,
                classical_model=classical_model,
                acoustic_features=features,
                speech_coverage=max(0.0, 1.0 - features.silence_ratio),
                audio_quality_score=quality.quality_score,
                audio_quality_verdict=quality.verdict,
            )
        except ExplainabilityError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FusionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    personal_baseline_response = None
    personalization_response = None
    if profile_id:
        baseline_service = PersonalBaselineService(settings.PERSONAL_BASELINE_DIR)
        if baseline_service.exists(profile_id):
            try:
                comparison = baseline_service.compare(
                    profile_id,
                    features.to_dict(),
                )
                personal_baseline_response = PersonalBaselineComparison(
                    profile_id=comparison.profile_id,
                    profile_version=comparison.profile_version,
                    sample_count=comparison.sample_count,
                    overall_deviation=comparison.overall_deviation,
                    normality=comparison.normality,
                    summary=comparison.summary,
                    metrics=[
                        PersonalBaselineMetricComparison(**metric.to_dict())
                        for metric in comparison.metrics
                    ],
                )
            except PersonalBaselineError as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc

        try:
            personalization = PersonalizedCalibrationService(settings.PERSONALIZATION_DIR).predict(
                profile_id,
                fused.probabilities,
                features.to_dict(),
            )
            personalization_response = PersonalizedPrediction(**personalization)
        except PersonalizationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    ordered = sorted(fused.probabilities.items(), key=lambda item: item[1], reverse=True)
    classical_probs = sorted(
        classical_prediction["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    deep_probs = sorted(
        deep_prediction.probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    classical_contribution = FusionModelContribution(
        model_type="classical_acoustic",
        model_name=classical_prediction["model_name"],
        emotion=max(
            classical_prediction["probabilities"],
            key=classical_prediction["probabilities"].get,
        ),
        confidence=classical_prediction["confidence"],
        calibrated_confidence=fused.classical_confidence,
        weight=fused.classical_weight,
        temperature=fused.classical_temperature,
        probabilities=[
            EmotionProbability(emotion=label, probability=value)
            for label, value in classical_probs
        ],
    )
    deep_contribution = FusionModelContribution(
        model_type="deep_speech",
        model_name=deep_prediction.metadata.model_id,
        emotion=deep_prediction.emotion,
        confidence=deep_prediction.confidence,
        calibrated_confidence=fused.deep_confidence,
        weight=fused.deep_weight,
        temperature=fused.deep_temperature,
        probabilities=[
            EmotionProbability(emotion=label, probability=value)
            for label, value in deep_probs
        ],
    )

    fusion_response = FusionEmotionResponse(
        success=True,
        message="Hybrid calibrated emotion fusion completed.",
        emotion=fused.emotion,
        confidence=fused.confidence,
        probabilities=[
            EmotionProbability(emotion=label, probability=value)
            for label, value in ordered
        ],
        fusion_strategy="temperature-calibrated weighted probability fusion",
        agreement=fused.agreement,
        js_divergence=fused.js_divergence,
        classical=classical_contribution,
        deep=deep_contribution,
        calibration_version=fused.calibration_version,
        sample_rate=processed.sample_rate,
        chunks_analyzed=deep_prediction.chunks_analyzed,
    )

    return AudioEmotionInsightsResponse(
        success=True,
        message="Emotion inference, acoustic evidence, and uncertainty analysis completed.",
        audio=_metadata_from_audio(processed),
        features=features_to_model(features),
        fusion=fusion_response,
        insights=EmotionInsights(
            summary=explanation.summary,
            evidence=[InsightEvidence(**item.to_dict()) for item in explanation.evidence],
            uncertainty=UncertaintyAnalysis(
                uncertainty_index=explanation.uncertainty.uncertainty_index,
                level=explanation.uncertainty.level,
                normalized_entropy=explanation.uncertainty.normalized_entropy,
                top_probability=explanation.uncertainty.top_probability,
                top_margin=explanation.uncertainty.top_margin,
                model_disagreement=explanation.uncertainty.model_disagreement,
                quality_score=explanation.uncertainty.quality_score,
                factors=[
                    UncertaintyFactor(**factor.to_dict())
                    for factor in explanation.uncertainty.factors
                ],
                recommendation=explanation.uncertainty.recommendation,
            ),
            alternatives=[
                InsightAlternative(emotion=emotion, probability=probability)
                for emotion, probability in explanation.alternatives
            ],
            caveat=explanation.caveat,
            personal_baseline=personal_baseline_response,
            personalization=personalization_response,
        ),
        audio_quality=AudioQualityReport(**quality.to_dict()),
    )


@router.get("/status")
def insights_status() -> dict:
    return {
        "available": True,
        "service": "emotion_insights",
        "message": "Evidence and uncertainty analysis is available when calibrated fusion is trained.",
    }


@router.post("", response_model=AudioEmotionInsightsResponse)
async def analyze_emotion_with_insights(
    file: UploadFile = File(...),
    profile_id: str | None = Form(default=None),
) -> AudioEmotionInsightsResponse:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported audio type. Upload a PCM WAV recording.",
        )
    try:
        audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    return _run_insights(audio_bytes, profile_id=profile_id)


def features_to_model(features):
    from app.models.features import AcousticFeatureSet

    return AcousticFeatureSet(**features.to_dict())
