from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.config import settings
from app.dimensional_emotion.model import DimensionalEmotionError
from app.dimensional_emotion.service import get_model, get_status
from app.affect_dynamics.engine import AffectDynamicsAnalyzer, AffectDynamicsError
from app.models.affect_dynamics import (
    AffectDynamicsEvent,
    AffectDelta,
    AffectDynamicsPoint,
    AffectDynamicsResponse,
    AffectState,
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


def _metadata_from_audio(processed):
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


def _state(values: dict[str, float] | None) -> AffectState | None:
    return AffectState(**values) if values is not None else None


@router.get("/status")
def affect_dynamics_status() -> dict:
    status = get_status()
    return {
        **status,
        "feature": "affect_dynamics",
        "window_seconds": 3.0,
        "hop_seconds": 1.0,
    }


@router.post("", response_model=AffectDynamicsResponse)
async def analyze_affect_dynamics(
    file: UploadFile = File(...),
) -> AffectDynamicsResponse:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio type. Upload a PCM WAV recording.")

    try:
        if not settings.DIMENSIONAL_EMOTION_ENABLED:
            raise HTTPException(status_code=503, detail="Dimensional affect analysis is disabled by configuration.")

        audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise AudioProcessingError("Audio file is too large. Maximum size is 12 MB.")

        processed = AudioProcessor().process(audio_bytes)
        if processed.is_silent:
            raise HTTPException(
                status_code=422,
                detail="Audio is nearly silent. Record a clear voice sample before affect-dynamics inference.",
            )

        try:
            model = get_model()
        except DimensionalEmotionError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        analysis = AffectDynamicsAnalyzer().analyze(
            processed.samples,
            processed.sample_rate,
            dimensional_model=model,
        )
    except AffectDynamicsError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()

    return AffectDynamicsResponse(
        success=True,
        message="Affect dynamics trajectory completed over overlapping speech windows with probability-free continuous state smoothing.",
        points=[
            AffectDynamicsPoint(
                index=point.index,
                start_seconds=point.start_seconds,
                end_seconds=point.end_seconds,
                duration_seconds=point.duration_seconds,
                speech_ratio=point.speech_ratio,
                is_speech=point.is_speech,
                raw=AffectState(**point.raw) if point.raw else None,
                smoothed=AffectState(**point.smoothed) if point.smoothed else None,
                deltas=AffectDelta(**point.deltas) if point.deltas else None,
                affect_shift=point.affect_shift,
                velocity=point.velocity,
                quadrant=point.quadrant,
            )
            for point in analysis.points
        ],
        events=[AffectDynamicsEvent(**event.to_dict()) for event in analysis.events],
        duration_seconds=analysis.duration_seconds,
        sample_rate=analysis.sample_rate,
        window_seconds=analysis.window_seconds,
        hop_seconds=analysis.hop_seconds,
        smoothing_alpha=analysis.smoothing_alpha,
        start_state=_state(analysis.start_state),
        end_state=_state(analysis.end_state),
        net_change=AffectDelta(**analysis.net_change),
        active_speech_seconds=analysis.active_speech_seconds,
        speech_coverage=analysis.speech_coverage,
        mean_velocity=analysis.mean_velocity,
        peak_velocity=analysis.peak_velocity,
        total_path_length=analysis.total_path_length,
        volatility=analysis.volatility,
        stability_score=analysis.stability_score,
        largest_shift=analysis.largest_shift,
        peak_arousal=analysis.peak_arousal,
        peak_arousal_at_seconds=analysis.peak_arousal_at_seconds,
        lowest_valence=analysis.lowest_valence,
        lowest_valence_at_seconds=analysis.lowest_valence_at_seconds,
        dominant_quadrant=analysis.dominant_quadrant,
        model_name=settings.DIMENSIONAL_EMOTION_MODEL,
        audio=_metadata_from_audio(processed),
    )
