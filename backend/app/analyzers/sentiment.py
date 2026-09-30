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

from app.analyzers.heuristic import heuristic_sentiment
from app.config import settings


class SentimentAnalyzer:
    def __init__(self) -> None:
        self.classifier = None
        self.mode = "heuristic"
        self.error: str | None = None

        if settings.MODEL_MODE == "heuristic":
            return

        if pipeline is None:
            self.error = "transformers is not installed; using heuristic sentiment scoring."
            return

        try:
            device = 0 if torch is not None and torch.cuda.is_available() else -1
            self.classifier = pipeline(
                task="text-classification",
                model=settings.SENTIMENT_MODEL,
                tokenizer=settings.SENTIMENT_MODEL,
                top_k=None,
                device=device,
            )
            self.mode = "transformer"
        except Exception as exc:
            self.error = f"sentiment model unavailable: {exc}"
            self.classifier = None

    def analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []

        if self.classifier is None:
            return heuristic_sentiment(texts)

        try:
            results = self.classifier(
                texts,
                batch_size=settings.MODEL_BATCH_SIZE,
                truncation=True,
            )
        except Exception as exc:
            self.error = f"sentiment inference failed: {exc}"
            self.classifier = None
            self.mode = "heuristic"
            return heuristic_sentiment(texts)

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

            negative = score_map.get("negative", 0.0)
            neutral = score_map.get("neutral", 0.0)
            positive = score_map.get("positive", 0.0)
            continuous_score = positive - negative
            dominant_label = max(score_map, key=score_map.get)

            output.append(
                {
                    "label": dominant_label,
                    "score": round(score_map.get(dominant_label, 0.0), 4),
                    "sentiment_score": round(continuous_score, 4),
                    "negative": round(negative, 4),
                    "neutral": round(neutral, 4),
                    "positive": round(positive, 4),
                }
            )

        return output
