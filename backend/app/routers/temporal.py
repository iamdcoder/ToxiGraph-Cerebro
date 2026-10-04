from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.config import settings
from app.deep_emotion.model import DeepEmotionModelError
from app.deep_emotion.service import get_model as get_deep_model
from app.emotion_baseline.model import BaselineModelError, ClassicalEmotionModel
from app.fusion.service import get_engine
from app.fusion.training import FusionTrainingError
from app.models.features import EmotionProbability
from app.models.temporal import (
    TemporalEmotionResponse,
    TemporalEmotionRun,
    TemporalEmotionSegment,
    TemporalEmotionTransition,
    TemporalTrajectoryPoint,
)
from app.temporal.engine import TemporalAnalysisError, TemporalConfig, TemporalEmotionAnalyzer


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
def temporal_status() -> dict:
    return {
        "available": True,
        "window_seconds": TemporalConfig().window_seconds,
        "hop_seconds": TemporalConfig().hop_seconds,
        "min_speech_ratio": TemporalConfig().min_speech_ratio,
        "smoothing_alpha": TemporalConfig().smoothing_alpha,
        "message": "Temporal emotion analysis is configured for overlapping speech windows.",
    }


@router.post("", response_model=TemporalEmotionResponse)
async def analyze_temporal_emotion(file: UploadFile = File(...)) -> TemporalEmotionResponse:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio type. Upload a PCM WAV recording.")

    try:
        try:
            classical_model = get_classical_model()
        except BaselineModelError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            deep_model = get_deep_model()
        except DeepEmotionModelError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            fusion_engine = get_engine()
        except FusionTrainingError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise AudioProcessingError("Audio file is too large. Maximum size is 12 MB.")

        processed = AudioProcessor().process(audio_bytes)
        if processed.is_silent:
            raise HTTPException(
                status_code=422,
                detail="Audio is nearly silent. Record a clear voice sample before temporal emotion analysis.",
            )

        analysis = TemporalEmotionAnalyzer().analyze(
            processed.samples,
            processed.sample_rate,
            classical_model=classical_model,
            deep_model=deep_model,
            fusion_engine=fusion_engine,
        )
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except TemporalAnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    segments = [
        TemporalEmotionSegment(
            index=item.index,
            start_seconds=item.start_seconds,
            end_seconds=item.end_seconds,
            duration_seconds=item.duration_seconds,
            speech_ratio=item.speech_ratio,
            is_speech=item.is_speech,
            raw_emotion=item.raw_emotion,
            raw_confidence=item.raw_confidence,
            emotion=item.emotion,
            confidence=item.confidence,
            probabilities=[
                EmotionProbability(emotion=label, probability=value)
                for label, value in sorted(item.probabilities.items(), key=lambda pair: pair[1], reverse=True)
            ],
            agreement=item.agreement,
            js_divergence=item.js_divergence,
            transition=item.transition,
            transition_score=item.transition_score,
        )
        for item in analysis.segments
    ]

    trajectory = [
        TemporalTrajectoryPoint(
            start_seconds=item.start_seconds,
            end_seconds=item.end_seconds,
            duration_seconds=item.duration_seconds,
            is_speech=item.is_speech,
            emotion=item.emotion,
            confidence=item.confidence,
            speech_ratio=item.speech_ratio,
            probabilities=[
                EmotionProbability(emotion=label, probability=value)
                for label, value in sorted(item.probabilities.items(), key=lambda pair: pair[1], reverse=True)
            ],
            covered_windows=item.covered_windows,
        )
        for item in analysis.trajectory
    ]

    runs = [
        TemporalEmotionRun(
            emotion=item.emotion,
            start_seconds=item.start_seconds,
            end_seconds=item.end_seconds,
            duration_seconds=item.duration_seconds,
            segment_count=item.segment_count,
            mean_confidence=item.mean_confidence,
        )
        for item in analysis.emotion_runs
    ]

    transitions = [
        TemporalEmotionTransition(
            from_emotion=item.from_emotion,
            to_emotion=item.to_emotion,
            at_seconds=item.at_seconds,
            score=item.score,
            source_segment_index=item.source_segment_index,
        )
        for item in analysis.transitions
    ]

    return TemporalEmotionResponse(
        success=True,
        message="Temporal emotion trajectory completed with overlapping speech windows and probability smoothing.",
        duration_seconds=analysis.duration_seconds,
        sample_rate=analysis.sample_rate,
        window_seconds=analysis.window_seconds,
        hop_seconds=analysis.hop_seconds,
        smoothing_alpha=analysis.smoothing_alpha,
        segments=segments,
        trajectory=trajectory,
        emotion_runs=runs,
        transitions=transitions,
        dominant_emotion=analysis.dominant_emotion,
        dominant_confidence=analysis.dominant_confidence,
        active_speech_seconds=analysis.active_speech_seconds,
        speech_coverage=analysis.speech_coverage,
        analyzed_windows=analysis.analyzed_windows,
        speech_windows=analysis.speech_windows,
        calibration_version=fusion_engine.artifact.version,
    )
