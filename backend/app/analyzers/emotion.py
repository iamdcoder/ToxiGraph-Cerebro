from __future__ import annotations

from typing import Any

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

from app.analyzers.heuristic import heuristic_emotion
from app.config import settings


EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


class EmotionAnalyzer:
    def __init__(self) -> None:
        self.classifier = None
        self.mode = "heuristic"
        self.error: str | None = None

        if settings.MODEL_MODE == "heuristic":
            return

        if pipeline is None:
            self.error = "transformers is not installed; using heuristic emotion scoring."
            return

        try:
            device = 0 if torch is not None and torch.cuda.is_available() else -1
            self.classifier = pipeline(
                task="text-classification",
                model=settings.EMOTION_MODEL,
                tokenizer=settings.EMOTION_MODEL,
                top_k=None,
                device=device,
            )
            self.mode = "transformer"
        except Exception as exc:
            self.error = f"emotion model unavailable: {exc}"
            self.classifier = None

    def analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []

        if self.classifier is None:
            return heuristic_emotion(texts)

        try:
            results = self.classifier(
                texts,
                batch_size=settings.MODEL_BATCH_SIZE,
                truncation=True,
            )
        except Exception as exc:
            self.error = f"emotion inference failed: {exc}"
            self.classifier = None
            self.mode = "heuristic"
            return heuristic_emotion(texts)

        if isinstance(results, dict):
            results = [results]

        output: list[dict[str, Any]] = []

        for scores in results:
            if isinstance(scores, dict):
                scores = [scores]

            score_map = {
                str(item["label"]).lower(): float(item["score"])
                for item in scores
            }

            emotion_scores = {
                emotion: round(score_map.get(emotion, 0.0), 4)
                for emotion in EMOTIONS
            }
            dominant_emotion = max(emotion_scores, key=emotion_scores.get)
            negative_emotion = (
                emotion_scores["anger"]
                + emotion_scores["disgust"]
                + emotion_scores["fear"]
                + emotion_scores["sadness"]
            )

            output.append(
                {
                    "dominant_emotion": dominant_emotion,
                    "scores": emotion_scores,
                    "negative_emotion_intensity": round(negative_emotion, 4),
                }
            )

        return output
