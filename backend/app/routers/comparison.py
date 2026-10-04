from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.comparison.engine import compare_insights
from app.config import settings
from app.models.comparison import (
    AudioComparisonResponse,
    ComparisonEmotionDelta,
    ComparisonFeatureDelta,
    ComparisonSummary,
)
from app.models.transcription import ComparisonContentVerification
from app.routers.insights import _run_insights
from app.transcription.service import ASRTranscriptionError, get_transcriber
from app.transcription.verification import verify_transcripts


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


async def _read(file: UploadFile) -> bytes:
    try:
        payload = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()
    if len(payload) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=422, detail="Audio file is too large. Maximum size is 12 MB.")
    return payload


@router.get("/status")
def comparison_status() -> dict:
    return {
        "available": True,
        "service": "recording_comparison",
        "message": "Compare two speech recordings through the same production inference stack.",
    }


@router.post("", response_model=AudioComparisonResponse)
async def compare_recordings(
    recording_a: UploadFile = File(...),
    recording_b: UploadFile = File(...),
    profile_id: str | None = Form(default=None),
) -> AudioComparisonResponse:
    _validate_content_type(recording_a)
    _validate_content_type(recording_b)

    raw_a = await _read(recording_a)
    raw_b = await _read(recording_b)

    try:
        analysis_a = _run_insights(raw_a, profile_id=profile_id)
        analysis_b = _run_insights(raw_b, profile_id=profile_id)
        result = compare_insights(analysis_a, analysis_b)

        content_verification: ComparisonContentVerification
        try:
            transcriber = get_transcriber()
            processed_a = AudioProcessor().process(raw_a)
            processed_b = AudioProcessor().process(raw_b)
            transcript_a = transcriber.transcribe(processed_a.samples, processed_a.sample_rate)
            transcript_b = transcriber.transcribe(processed_b.samples, processed_b.sample_rate)
            verification = verify_transcripts(transcript_a.text, transcript_b.text)
            content_verification = ComparisonContentVerification(
                status=verification.status,
                similarity=verification.similarity,
                word_error_rate=verification.word_error_rate,
                words_a=verification.words_a,
                words_b=verification.words_b,
                transcript_a=transcript_a.text,
                transcript_b=transcript_b.text,
                normalized_a=verification.normalized_a,
                normalized_b=verification.normalized_b,
                interpretation=verification.interpretation,
                method=verification.method,
                asr_model_id=transcript_a.model_id,
                asr_available=True,
            )
        except ASRTranscriptionError as exc:
            content_verification = ComparisonContentVerification(
                status="unavailable",
                similarity=0.0,
                word_error_rate=0.0,
                words_a=0,
                words_b=0,
                transcript_a="",
                transcript_b="",
                normalized_a="",
                normalized_b="",
                interpretation=(
                    "Local speech transcription is unavailable, so CEREBRO cannot verify whether the two recordings used identical words. "
                    f"The acoustic and emotion comparison remains available. Reason: {exc}"
                ),
                method="local Whisper transcription unavailable",
                asr_model_id=settings.ASR_MODEL or None,
                asr_available=False,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return AudioComparisonResponse(
        success=True,
        message="Two recordings were analyzed through the same calibrated CEREBRO pipeline and their spoken content was checked when local ASR was available.",
        protocol_note=(
            "For the cleanest tone-comparison experiment, speak the same sentence in both recordings and change only the delivery. "
            "The content verifier compares normalized ASR word sequences and returns same, different, uncertain, or unavailable."
        ),
        recording_a=analysis_a,
        recording_b=analysis_b,
        comparison=ComparisonSummary(**result.to_dict()),
        acoustic_features=[ComparisonFeatureDelta(**item.to_dict()) for item in result.acoustic_features],
        emotion_probabilities=[ComparisonEmotionDelta(**item.to_dict()) for item in result.emotion_probabilities],
        content_verification=content_verification,
    )
