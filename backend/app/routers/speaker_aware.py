from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.conversation_graph.engine import build_conversation_graph, ConversationGraphError
from app.conversation_graph.dynamics import ConversationDynamicsAnalyzer, ConversationDynamicsError
from app.conversation_state.engine import ConversationStateAnalyzer, ConversationStateError
from app.config import settings
from app.dimensional_emotion.model import DimensionalEmotionError
from app.dimensional_emotion.service import get_model as get_affect_model
from app.emotion_baseline.model import BaselineModelError, ClassicalEmotionModel
from app.fusion.engine import FusionError
from app.fusion.service import get_engine
from app.fusion.training import FusionTrainingError
from app.deep_emotion.model import DeepEmotionModelError
from app.deep_emotion.service import get_model as get_deep_model
from app.models.audio import AudioMetadata
from app.models.features import EmotionProbability
from app.models.speaker_analysis import (
    ConversationGraphEdgeResponse,
    ConversationGraphMetricsResponse,
    ConversationGraphNodeResponse,
    ConversationInteractionGraphResponse,
    ConversationDynamicsResponse,
    ConversationDynamicsEventResponse,
    ConversationPairDynamicsResponse,
    ConversationStateResponse,
    SpeakerAwareResponse,
    SpeakerInteractionResponse,
    SpeakerSummaryResponse,
    SpeakerTurnResponse,
)
from app.speaker_diarization.engine import SpeakerAwareAnalyzer, SpeakerAnalysisError
from app.models.conversation_state import ConversationStatePointResponse, ConversationStateTransitionResponse
from app.speaker_diarization.model import SpeakerDiarizationError
from app.speaker_diarization.service import get_model as get_diarization_model, get_status

router = APIRouter()


@router.get("/status")
def speaker_aware_status() -> dict:
    status = get_status()
    return {
        **status,
        "feature": "speaker_aware_emotion",
        "minimum_turn_seconds": settings.SPEAKER_DIARIZATION_MIN_TURN_SECONDS,
        "max_speakers": settings.SPEAKER_DIARIZATION_MAX_SPEAKERS,
    }


def _audio_metadata(processed) -> AudioMetadata:
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


def _predictor_factory():
    try:
        fusion_engine = get_engine()
    except FusionTrainingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        classical_model = ClassicalEmotionModel.load(settings.CLASSICAL_MODEL_PATH)
    except BaselineModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    try:
        deep_model = get_deep_model()
    except DeepEmotionModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    affect_model = None
    if settings.DIMENSIONAL_EMOTION_ENABLED:
        try:
            affect_model = get_affect_model()
        except DimensionalEmotionError:
            affect_model = None

    def predict(samples, sample_rate):
        from app.features.acoustic import extract_acoustic_features

        features = extract_acoustic_features(samples, sample_rate)
        classical = classical_model.predict(features.to_vector())
        deep = deep_model.predict(samples, sample_rate)
        fused = fusion_engine.combine(classical["probabilities"], deep.probabilities)
        output = {
            "emotion": fused.emotion,
            "confidence": fused.confidence,
            "probabilities": fused.probabilities,
        }
        if affect_model is not None:
            try:
                affect = affect_model.predict(samples, sample_rate)
                output.update(affect.values)
            except DimensionalEmotionError:
                pass
        return output

    return predict, fusion_engine.artifact.version, affect_model.metadata()["model_id"] if affect_model else None


@router.post("", response_model=SpeakerAwareResponse)
async def analyze_speakers(
    file: UploadFile = File(...),
) -> SpeakerAwareResponse:
    content_type = (file.content_type or "").lower()
    allowed = {
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/vnd.wave",
        "application/octet-stream",
    }
    if content_type and content_type not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported audio type. Upload a PCM WAV recording.")

    try:
        if not settings.SPEAKER_DIARIZATION_ENABLED:
            raise HTTPException(status_code=503, detail="Speaker-aware analysis is disabled by configuration.")
        audio_bytes = await file.read(settings.SPEAKER_DIARIZATION_MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > settings.SPEAKER_DIARIZATION_MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=422, detail="Audio file is too large for speaker-aware analysis.")
        processed = AudioProcessor().process(audio_bytes)
        if processed.duration_seconds > settings.SPEAKER_DIARIZATION_MAX_DURATION_SECONDS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Audio is too long for speaker-aware analysis. Maximum duration is "
                    f"{settings.SPEAKER_DIARIZATION_MAX_DURATION_SECONDS:g} seconds."
                ),
            )
        if processed.is_silent:
            raise HTTPException(status_code=422, detail="Audio is nearly silent. Record a multi-speaker conversation with clear speech.")

        try:
            diarizer = get_diarization_model()
        except SpeakerDiarizationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            turns = diarizer.diarize(processed.samples, processed.sample_rate)
        except SpeakerDiarizationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        predictor, calibration_version, affect_model_name = _predictor_factory()
        analysis = SpeakerAwareAnalyzer().analyze(
            processed.samples,
            processed.sample_rate,
            turns,
            predictor=predictor,
            min_turn_seconds=settings.SPEAKER_DIARIZATION_MIN_TURN_SECONDS,
        )
        try:
            interaction_graph = build_conversation_graph(analysis)
            conversation_dynamics = ConversationDynamicsAnalyzer().analyze(analysis)
            conversation_state = ConversationStateAnalyzer().analyze(conversation_dynamics, duration_seconds=analysis.duration_seconds)
        except ConversationGraphError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ConversationDynamicsError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ConversationStateError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FusionError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except SpeakerAnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    payload = analysis.to_dict()
    graph_payload = interaction_graph.to_dict()
    return SpeakerAwareResponse(
        success=True,
        message=(
            "Speaker-aware emotion analysis completed. Each diarized speaker turn was analyzed independently, "
            "then summarized into speaker-level emotion and affect trajectories."
        ),
        duration_seconds=payload["duration_seconds"],
        sample_rate=payload["sample_rate"],
        speaker_count=payload["speaker_count"],
        speakers=[SpeakerSummaryResponse(**item) for item in payload["speakers"]],
        turns=[
            SpeakerTurnResponse(
                **{**item, "probabilities": [EmotionProbability(emotion=key, probability=value) for key, value in item["probabilities"].items()]}
            )
            for item in payload["turns"]
        ],
        interactions=[SpeakerInteractionResponse(**item) for item in payload["interactions"]],
        total_speaking_seconds=payload["total_speaking_seconds"],
        speech_coverage=payload["speech_coverage"],
        overlap_seconds=payload["overlap_seconds"],
        speaker_switches=payload["speaker_switches"],
        dominant_speaker=payload["dominant_speaker"],
        diarization_model=diarizer.model_id,
        diarization_metadata=diarizer.metadata(),
        emotion_model=f"fusion:{calibration_version}",
        affect_model=affect_model_name,
        audio=_audio_metadata(processed),
        interaction_graph=ConversationInteractionGraphResponse(
            graph_type=graph_payload["graph_type"],
            directed=graph_payload["directed"],
            nodes=[ConversationGraphNodeResponse(**item) for item in graph_payload["nodes"]],
            edges=[ConversationGraphEdgeResponse(**item) for item in graph_payload["edges"]],
            metrics=ConversationGraphMetricsResponse(**graph_payload["metrics"]),
            semantics=graph_payload["semantics"],
        ),
        conversation_dynamics=ConversationDynamicsResponse(
            **conversation_dynamics.to_dict() | {
                "events": [ConversationDynamicsEventResponse(**item) for item in conversation_dynamics.to_dict()["events"]],
                "pair_metrics": [ConversationPairDynamicsResponse(**item) for item in conversation_dynamics.to_dict()["pair_metrics"]],
            }
        ),
        conversation_state=ConversationStateResponse(
            **conversation_state.to_dict() | {
                "points": [ConversationStatePointResponse(**item) for item in conversation_state.to_dict()["points"]],
                "transitions": [ConversationStateTransitionResponse(**item) for item in conversation_state.to_dict()["transitions"]],
            }
        ),
    )
