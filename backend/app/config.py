import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "CEREBRO")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.2.0")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

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
    # to deterministic local heuristics so the demo remains usable.
    MODEL_MODE: str = os.getenv("MODEL_MODE", "auto").strip().lower()


settings = Settings()
