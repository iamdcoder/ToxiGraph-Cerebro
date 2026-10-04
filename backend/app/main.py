from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.privacy.service import PrivacyService
from app.security import SecurityMiddleware
from app.routers import affect, affect_dynamics, analysis, audio, comparison, cross_session_patterns, deep_emotion, emotion, evaluation, features, fusion, health, insights, live, live_multi_speaker, longitudinal_emotion, optimization, personal_baseline, personalization, privacy, quality, robustness, speaker_aware, temporal, threads, transcription, voice_change


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "CEREBRO — Speech Emotion Intelligence. "
        "A staged system for analyzing emotion conveyed through speech "
        "using measurable acoustic and learned speech representations."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)


app.add_middleware(
    SecurityMiddleware,
    enabled=settings.SECURITY_HEADERS_ENABLED,
    max_body_bytes=settings.SECURITY_MAX_BODY_BYTES,
    rate_limit_enabled=settings.HTTP_RATE_LIMIT_ENABLED,
    rate_limit_requests=settings.HTTP_RATE_LIMIT_REQUESTS,
    rate_limit_window_seconds=settings.HTTP_RATE_LIMIT_WINDOW_SECONDS,
    api_auth_enabled=settings.API_AUTH_ENABLED,
    api_key=settings.API_AUTH_KEY,
    hsts_enabled=settings.SECURITY_HSTS_ENABLED,
)


app.include_router(
    health.router,
    prefix=f"{settings.API_PREFIX}/health",
    tags=["Health"],
)

app.include_router(
    threads.router,
    prefix=f"{settings.API_PREFIX}/threads",
    tags=["Threads"],
)

app.include_router(
    analysis.router,
    prefix=f"{settings.API_PREFIX}/analysis",
    tags=["Analysis"],
)

app.include_router(
    audio.router,
    prefix=f"{settings.API_PREFIX}/audio",
    tags=["Audio"],
)

app.include_router(
    features.router,
    prefix=f"{settings.API_PREFIX}/audio/features",
    tags=["Acoustic Features"],
)

app.include_router(
    emotion.router,
    prefix=f"{settings.API_PREFIX}/audio/emotion",
    tags=["Emotion Inference"],
)

app.include_router(
    deep_emotion.router,
    prefix=f"{settings.API_PREFIX}/audio/emotion/deep",
    tags=["Deep Speech Emotion Inference"],
)

app.include_router(
    fusion.router,
    prefix=f"{settings.API_PREFIX}/audio/emotion/fusion",
    tags=["Hybrid Emotion Fusion"],
)

app.include_router(
    temporal.router,
    prefix=f"{settings.API_PREFIX}/audio/emotion/temporal",
    tags=["Temporal Emotion Analysis"],
)

app.include_router(
    affect.router,
    prefix=f"{settings.API_PREFIX}/audio/emotion/affect",
    tags=["Dimensional Speech Affect"],
)

app.include_router(
    affect_dynamics.router,
    prefix=f"{settings.API_PREFIX}/audio/emotion/affect/dynamics",
    tags=["Affect Dynamics"],
)

app.include_router(
    insights.router,
    prefix=f"{settings.API_PREFIX}/audio/emotion/insights",
    tags=["Emotion Insights"],
)

app.include_router(
    personal_baseline.router,
    prefix=f"{settings.API_PREFIX}/audio/baseline",
    tags=["Personal Voice Baseline"],
)

app.include_router(
    personalization.router,
    prefix=f"{settings.API_PREFIX}/audio/personalization",
    tags=["Personalized Emotion Calibration"],
)

app.include_router(
    longitudinal_emotion.router,
    prefix=f"{settings.API_PREFIX}/audio/history",
    tags=["Longitudinal Voice Intelligence"],
)

app.include_router(
    cross_session_patterns.router,
    prefix=f"{settings.API_PREFIX}/audio/history-patterns",
    tags=["Cross-Session Patterns"],
)

app.include_router(
    voice_change.router,
    prefix=f"{settings.API_PREFIX}/audio/voice-change",
    tags=["Voice Change Detection"],
)

app.include_router(
    comparison.router,
    prefix=f"{settings.API_PREFIX}/audio/emotion/compare",
    tags=["Recording Comparison"],
)

app.include_router(
    transcription.router,
    prefix=f"{settings.API_PREFIX}/audio/transcribe",
    tags=["Speech Transcription"],
)

app.include_router(
    quality.router,
    prefix=f"{settings.API_PREFIX}/audio/quality",
    tags=["Audio Quality"],
)

app.include_router(
    evaluation.router,
    prefix=f"{settings.API_PREFIX}/evaluation",
    tags=["Evaluation"],
)

app.include_router(
    optimization.router,
    prefix=f"{settings.API_PREFIX}/evaluation/optimization",
    tags=["Model Optimization"],
)

app.include_router(
    robustness.router,
    prefix=f"{settings.API_PREFIX}/robustness",
    tags=["Robustness"],
)

app.include_router(
    privacy.router,
    prefix=f"{settings.API_PREFIX}/privacy",
    tags=["Privacy & Data Controls"],
)


app.include_router(
    live.router,
    prefix=f"{settings.API_PREFIX}/audio/live",
    tags=["Live Speech Emotion"],
)

app.include_router(
    speaker_aware.router,
    prefix=f"{settings.API_PREFIX}/audio/speakers",
    tags=["Speaker-Aware Emotion Analysis"],
)


@app.get("/")
def root() -> dict:
    return {
        "name": "CEREBRO",
        "codename": "CEREBRO",
        "version": settings.APP_VERSION,
        "message": "Speech Emotion Intelligence API is running.",
    }
app.include_router(
    live_multi_speaker.router,
    prefix=f"{settings.API_PREFIX}/audio/live-multi",
    tags=["Live Multi-Speaker Emotion"],
)


PrivacyService(
    personal_baseline_dir=settings.PERSONAL_BASELINE_DIR,
    voice_history_dir=settings.VOICE_HISTORY_DIR,
    personalization_dir=settings.PERSONALIZATION_DIR,
    retention_days=settings.PRIVACY_RETENTION_DAYS,
    retention_enabled=settings.PRIVACY_RETENTION_ENABLED,
).prune_expired()
