import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "CEREBRO")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.27.0")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    SECURITY_HEADERS_ENABLED: bool = os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true"
    SECURITY_MAX_BODY_BYTES: int = int(os.getenv("SECURITY_MAX_BODY_BYTES", str(20 * 1024 * 1024)))
    HTTP_RATE_LIMIT_ENABLED: bool = os.getenv("HTTP_RATE_LIMIT_ENABLED", "true").lower() == "true"
    HTTP_RATE_LIMIT_REQUESTS: int = int(os.getenv("HTTP_RATE_LIMIT_REQUESTS", "120"))
    HTTP_RATE_LIMIT_WINDOW_SECONDS: float = float(os.getenv("HTTP_RATE_LIMIT_WINDOW_SECONDS", "60"))
    API_AUTH_ENABLED: bool = os.getenv("API_AUTH_ENABLED", "false").lower() == "true"
    API_AUTH_KEY: str | None = os.getenv("API_AUTH_KEY", "").strip() or None
    SECURITY_HSTS_ENABLED: bool = os.getenv("SECURITY_HSTS_ENABLED", "false").lower() == "true"
    WEBSOCKET_AUTH_ENABLED: bool = os.getenv("WEBSOCKET_AUTH_ENABLED", "false").lower() == "true"
    WEBSOCKET_AUTH_TOKEN: str | None = os.getenv("WEBSOCKET_AUTH_TOKEN", "").strip() or None
    WEBSOCKET_MAX_CONNECTIONS_PER_IP: int = int(os.getenv("WEBSOCKET_MAX_CONNECTIONS_PER_IP", "3"))

    PRIVACY_RETENTION_ENABLED: bool = os.getenv("PRIVACY_RETENTION_ENABLED", "true").lower() == "true"
    PRIVACY_RETENTION_DAYS: int = int(os.getenv("PRIVACY_RETENTION_DAYS", "30"))

    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")

    ALLOWED_ORIGINS: list[str] = [
        origin.strip()
        for origin in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:5173,http://localhost:3000",
        ).split(",")
        if origin.strip()
    ]

    SENTIMENT_MODEL: str = os.getenv(
        "SENTIMENT_MODEL",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    )

    TOXICITY_MODEL: str = os.getenv(
        "TOXICITY_MODEL",
        "unitary/unbiased-toxic-roberta",
    )

    EMOTION_MODEL: str = os.getenv(
        "EMOTION_MODEL",
        "j-hartmann/emotion-english-distilroberta-base",
    )

    MODEL_BATCH_SIZE: int = int(os.getenv("MODEL_BATCH_SIZE", "4"))

    # auto = use Hugging Face models when available, otherwise fall back
    # to deterministic local heuristics so local development remains usable.
    MODEL_MODE: str = os.getenv("MODEL_MODE", "auto").strip().lower()

    CLASSICAL_MODEL_PATH: str = os.getenv(
        "CLASSICAL_MODEL_PATH",
        "models/cerebro_classical_emotion.joblib",
    )

    DEEP_EMOTION_MODEL: str = os.getenv(
        "DEEP_EMOTION_MODEL",
        "Dpngtm/wav2vec2-emotion-recognition",
    )

    DEEP_EMOTION_DEVICE: str = os.getenv("DEEP_EMOTION_DEVICE", "auto").strip().lower()
    DEEP_EMOTION_SAMPLE_RATE: int = int(os.getenv("DEEP_EMOTION_SAMPLE_RATE", "16000"))
    DEEP_EMOTION_MAX_CHUNK_SECONDS: float = float(os.getenv("DEEP_EMOTION_MAX_CHUNK_SECONDS", "10"))
    DEEP_EMOTION_CHUNK_SECONDS: float = float(os.getenv("DEEP_EMOTION_CHUNK_SECONDS", "8"))
    DEEP_EMOTION_CHUNK_OVERLAP_SECONDS: float = float(os.getenv("DEEP_EMOTION_CHUNK_OVERLAP_SECONDS", "1"))

    FUSION_CALIBRATION_PATH: str = os.getenv(
        "FUSION_CALIBRATION_PATH",
        "models/cerebro_fusion_calibration.joblib",
    )

    ASR_MODEL: str = os.getenv(
        "ASR_MODEL",
        "openai/whisper-small",
    )
    ASR_DEVICE: str = os.getenv("ASR_DEVICE", "auto").strip().lower()
    ASR_SAMPLE_RATE: int = int(os.getenv("ASR_SAMPLE_RATE", "16000"))
    ASR_LANGUAGE: str | None = os.getenv("ASR_LANGUAGE", "").strip() or None
    ASR_MAX_NEW_TOKENS: int = int(os.getenv("ASR_MAX_NEW_TOKENS", "128"))

    DIMENSIONAL_EMOTION_ENABLED: bool = os.getenv("DIMENSIONAL_EMOTION_ENABLED", "true").lower() == "true"
    DIMENSIONAL_EMOTION_MODEL: str = os.getenv(
        "DIMENSIONAL_EMOTION_MODEL",
        "3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes",
    )
    DIMENSIONAL_EMOTION_DEVICE: str = os.getenv("DIMENSIONAL_EMOTION_DEVICE", "auto").strip().lower()
    DIMENSIONAL_EMOTION_SAMPLE_RATE: int = int(os.getenv("DIMENSIONAL_EMOTION_SAMPLE_RATE", "16000"))
    DIMENSIONAL_EMOTION_MAX_CHUNK_SECONDS: float = float(os.getenv("DIMENSIONAL_EMOTION_MAX_CHUNK_SECONDS", "30"))
    DIMENSIONAL_EMOTION_CHUNK_SECONDS: float = float(os.getenv("DIMENSIONAL_EMOTION_CHUNK_SECONDS", "15"))
    DIMENSIONAL_EMOTION_CHUNK_OVERLAP_SECONDS: float = float(os.getenv("DIMENSIONAL_EMOTION_CHUNK_OVERLAP_SECONDS", "3"))

    PERSONAL_BASELINE_DIR: str = os.getenv(
        "PERSONAL_BASELINE_DIR",
        "artifacts/personal_baselines",
    )

    VOICE_HISTORY_DIR: str = os.getenv(
        "VOICE_HISTORY_DIR",
        "artifacts/voice_history",
    )

    EVALUATION_REPORT_PATH: str = os.getenv(
        "EVALUATION_REPORT_PATH",
        "artifacts/evaluation/evaluation.json",
    )

    OPTIMIZATION_REPORT_PATH: str = os.getenv(
        "OPTIMIZATION_REPORT_PATH",
        "artifacts/optimization/classical/optimization.json",
    )

    ROBUSTNESS_REPORT_PATH: str = os.getenv(
        "ROBUSTNESS_REPORT_PATH",
        "artifacts/robustness/robustness.json",
    )

    PERSONALIZATION_DIR: str = os.getenv(
        "PERSONALIZATION_DIR",
        "artifacts/personalization",
    )

    LIVE_STREAM_ENABLED: bool = os.getenv("LIVE_STREAM_ENABLED", "true").lower() == "true"
    LIVE_STREAM_TARGET_SAMPLE_RATE: int = int(os.getenv("LIVE_STREAM_TARGET_SAMPLE_RATE", "16000"))
    LIVE_STREAM_WINDOW_SECONDS: float = float(os.getenv("LIVE_STREAM_WINDOW_SECONDS", "3.0"))
    LIVE_STREAM_HOP_SECONDS: float = float(os.getenv("LIVE_STREAM_HOP_SECONDS", "1.0"))
    LIVE_STREAM_MIN_WINDOW_SECONDS: float = float(os.getenv("LIVE_STREAM_MIN_WINDOW_SECONDS", "1.5"))
    LIVE_STREAM_MAX_DURATION_SECONDS: float = float(os.getenv("LIVE_STREAM_MAX_DURATION_SECONDS", "120"))
    LIVE_STREAM_MAX_CHUNK_BYTES: int = int(os.getenv("LIVE_STREAM_MAX_CHUNK_BYTES", str(256 * 1024)))
    LIVE_STREAM_MIN_SPEECH_RATIO: float = float(os.getenv("LIVE_STREAM_MIN_SPEECH_RATIO", "0.30"))
    LIVE_STREAM_SMOOTHING_ALPHA: float = float(os.getenv("LIVE_STREAM_SMOOTHING_ALPHA", "0.35"))


    LIVE_MULTI_SPEAKER_ENABLED: bool = os.getenv("LIVE_MULTI_SPEAKER_ENABLED", "true").lower() == "true"
    LIVE_MULTI_SPEAKER_TARGET_SAMPLE_RATE: int = int(os.getenv("LIVE_MULTI_SPEAKER_TARGET_SAMPLE_RATE", "16000"))
    LIVE_MULTI_SPEAKER_CONTEXT_SECONDS: float = float(os.getenv("LIVE_MULTI_SPEAKER_CONTEXT_SECONDS", "8.0"))
    LIVE_MULTI_SPEAKER_HOP_SECONDS: float = float(os.getenv("LIVE_MULTI_SPEAKER_HOP_SECONDS", "1.0"))
    LIVE_MULTI_SPEAKER_MIN_CONTEXT_SECONDS: float = float(os.getenv("LIVE_MULTI_SPEAKER_MIN_CONTEXT_SECONDS", "3.0"))
    LIVE_MULTI_SPEAKER_STABILIZATION_SECONDS: float = float(os.getenv("LIVE_MULTI_SPEAKER_STABILIZATION_SECONDS", "0.8"))
    LIVE_MULTI_SPEAKER_MAX_DURATION_SECONDS: float = float(os.getenv("LIVE_MULTI_SPEAKER_MAX_DURATION_SECONDS", "180"))
    LIVE_MULTI_SPEAKER_MATCH_THRESHOLD: float = float(os.getenv("LIVE_MULTI_SPEAKER_MATCH_THRESHOLD", "0.18"))

    SPEAKER_DIARIZATION_ENABLED: bool = os.getenv("SPEAKER_DIARIZATION_ENABLED", "true").lower() == "true"
    SPEAKER_DIARIZATION_MODEL: str = os.getenv(
        "SPEAKER_DIARIZATION_MODEL",
        "pyannote/speaker-diarization-community-1",
    )
    PYANNOTE_TOKEN: str | None = os.getenv("PYANNOTE_TOKEN", "").strip() or None
    SPEAKER_DIARIZATION_DEVICE: str = os.getenv("SPEAKER_DIARIZATION_DEVICE", "auto").strip().lower()
    SPEAKER_DIARIZATION_SAMPLE_RATE: int = int(os.getenv("SPEAKER_DIARIZATION_SAMPLE_RATE", "16000"))
    SPEAKER_DIARIZATION_MIN_SPEAKERS: int | None = (
        int(os.getenv("SPEAKER_DIARIZATION_MIN_SPEAKERS"))
        if os.getenv("SPEAKER_DIARIZATION_MIN_SPEAKERS")
        else None
    )
    SPEAKER_DIARIZATION_MAX_SPEAKERS: int | None = (
        int(os.getenv("SPEAKER_DIARIZATION_MAX_SPEAKERS"))
        if os.getenv("SPEAKER_DIARIZATION_MAX_SPEAKERS")
        else 6
    )
    SPEAKER_DIARIZATION_MIN_TURN_SECONDS: float = float(
        os.getenv("SPEAKER_DIARIZATION_MIN_TURN_SECONDS", "0.8")
    )
    SPEAKER_DIARIZATION_MAX_UPLOAD_BYTES: int = int(
        os.getenv("SPEAKER_DIARIZATION_MAX_UPLOAD_BYTES", str(16 * 1024 * 1024))
    )
    SPEAKER_DIARIZATION_MAX_DURATION_SECONDS: float = float(
        os.getenv("SPEAKER_DIARIZATION_MAX_DURATION_SECONDS", "180")
    )


settings = Settings()
