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

from app.analyzers.heuristic import heuristic_toxicity
from app.config import settings


PRIMARY_TOXICITY_LABELS = {
    "toxicity",
    "severe_toxicity",
    "obscene",
    "identity_attack",
    "insult",
    "threat",
    "sexual_explicit",
}


class ToxicityAnalyzer:
    def __init__(self) -> None:
        self.classifier = None
        self.mode = "heuristic"
        self.error: str | None = None

        if settings.MODEL_MODE == "heuristic":
            return

        if pipeline is None:
            self.error = "transformers is not installed; using heuristic toxicity scoring."
            return

        try:
            device = 0 if torch is not None and torch.cuda.is_available() else -1
            self.classifier = pipeline(
                task="text-classification",
                model=settings.TOXICITY_MODEL,
                tokenizer=settings.TOXICITY_MODEL,
                top_k=None,
                device=device,
            )
            self.mode = "transformer"
        except Exception as exc:
            self.error = f"toxicity model unavailable: {exc}"
            self.classifier = None

    def _risk_level(self, toxicity: float) -> str:
        if toxicity >= 0.70:
            return "critical"
        if toxicity >= 0.50:
            return "high"
        if toxicity >= 0.25:
            return "elevated"
        return "low"

    def analyze(self, texts: list[str]) -> list[dict[str, Any]]:
        if not texts:
            return []

        if self.classifier is None:
            return heuristic_toxicity(texts)

        try:
            results = self.classifier(
                texts,
                batch_size=settings.MODEL_BATCH_SIZE,
                truncation=True,
            )
        except Exception as exc:
            self.error = f"toxicity inference failed: {exc}"
            self.classifier = None
            self.mode = "heuristic"
            return heuristic_toxicity(texts)

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

            toxicity = score_map.get("toxicity", 0.0)
            primary_scores = {
                label: score_map.get(label, 0.0)
                for label in PRIMARY_TOXICITY_LABELS
            }
            strongest_signal = max(primary_scores, key=primary_scores.get)

            output.append(
                {
                    "toxicity": round(toxicity, 4),
                    "severe_toxicity": round(score_map.get("severe_toxicity", 0.0), 4),
                    "obscene": round(score_map.get("obscene", 0.0), 4),
                    "identity_attack": round(score_map.get("identity_attack", 0.0), 4),
                    "insult": round(score_map.get("insult", 0.0), 4),
                    "threat": round(score_map.get("threat", 0.0), 4),
                    "sexual_explicit": round(score_map.get("sexual_explicit", 0.0), 4),
                    "strongest_signal": strongest_signal,
                    "risk_level": self._risk_level(toxicity),
                }
            )

        return output
