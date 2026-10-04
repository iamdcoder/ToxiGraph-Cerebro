from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.audio.processor import AudioProcessingError, AudioProcessor
from app.config import settings
from app.features.acoustic import extract_acoustic_features
from app.models.personal_baseline import (
    PersonalBaselineComparison,
    PersonalBaselineCreateResponse,
    PersonalBaselineMetricComparison,
    PersonalBaselineStatus,
)
from app.personal_baseline.service import (
    MAX_SAMPLES,
    MIN_SAMPLES,
    PersonalBaselineError,
    PersonalBaselineService,
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


def get_service() -> PersonalBaselineService:
    return PersonalBaselineService(settings.PERSONAL_BASELINE_DIR)


def _status(profile_id: str, service: PersonalBaselineService) -> PersonalBaselineStatus:
    try:
        profile = service.load(profile_id)
    except PersonalBaselineError as exc:
        return PersonalBaselineStatus(
            available=False,
            profile_id=profile_id,
            sample_count=0,
            reference_duration_seconds=0,
            message=str(exc),
        )
    return PersonalBaselineStatus(
        available=True,
        profile_id=profile.profile_id,
        version=profile.version,
        sample_count=profile.sample_count,
        reference_duration_seconds=profile.reference_duration_seconds,
        message="Personal acoustic baseline is ready.",
    )


def _check_type(file: UploadFile) -> None:
    content_type = (file.content_type or "").lower()
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported audio type. Upload a PCM WAV recording.")


async def _read_features(file: UploadFile) -> dict:
    _check_type(file)
    try:
        audio_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > MAX_UPLOAD_BYTES:
            raise AudioProcessingError("Audio file is too large. Maximum size is 12 MB.")
        processed = AudioProcessor().process(audio_bytes)
        if processed.is_silent:
            raise AudioProcessingError("Baseline samples cannot be nearly silent. Record a clear voice sample.")
        features = extract_acoustic_features(processed.samples, processed.sample_rate)
        return features.to_dict()
    finally:
        await file.close()


@router.get("/status")
def baseline_service_status() -> dict:
    return {
        "available": True,
        "service": "personal_baseline",
        "min_samples": MIN_SAMPLES,
        "max_samples": MAX_SAMPLES,
        "message": "Personal baseline compares current acoustic behavior with a speaker's own reference samples.",
    }


@router.get("/{profile_id}", response_model=PersonalBaselineStatus)
def get_baseline(profile_id: str) -> PersonalBaselineStatus:
    service = get_service()
    try:
        return _status(profile_id, service)
    except PersonalBaselineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/{profile_id}", response_model=PersonalBaselineCreateResponse)
async def create_baseline(
    profile_id: str,
    files: list[UploadFile] = File(...),
) -> PersonalBaselineCreateResponse:
    if not MIN_SAMPLES <= len(files) <= MAX_SAMPLES:
        for file in files:
            await file.close()
        raise HTTPException(
            status_code=422,
            detail=f"Upload {MIN_SAMPLES}–{MAX_SAMPLES} reference samples to build a personal baseline.",
        )

    service = get_service()
    feature_payloads: list[dict] = []
    try:
        for file in files:
            feature_payloads.append(await _read_features(file))
        profile = service.create(profile_id, feature_payloads)
    except (AudioProcessingError, PersonalBaselineError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        for file in files:
            await file.close()

    return PersonalBaselineCreateResponse(
        success=True,
        message=f"Personal baseline created from {profile.sample_count} reference samples.",
        profile=PersonalBaselineStatus(
            available=True,
            profile_id=profile.profile_id,
            version=profile.version,
            sample_count=profile.sample_count,
            reference_duration_seconds=profile.reference_duration_seconds,
            message="Personal acoustic baseline is ready.",
        ),
    )


@router.delete("/{profile_id}")
def delete_baseline(profile_id: str) -> dict:
    try:
        get_service().delete(profile_id)
    except PersonalBaselineError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"success": True, "message": "Personal baseline deleted."}


@router.post("/{profile_id}/compare", response_model=PersonalBaselineComparison)
async def compare_with_baseline(
    profile_id: str,
    file: UploadFile = File(...),
) -> PersonalBaselineComparison:
    try:
        features = await _read_features(file)
        comparison = get_service().compare(profile_id, features)
    except PersonalBaselineError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except AudioProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return PersonalBaselineComparison(
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
