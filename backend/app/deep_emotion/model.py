from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


STANDARD_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised",
)

LABEL_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "calm": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
}


class DeepEmotionModelError(RuntimeError):
    """Raised when deep speech-emotion inference cannot be performed."""


@dataclass(frozen=True)
class DeepEmotionMetadata:
    model_id: str
    labels: tuple[str, ...]
    sample_rate: int
    device: str


@dataclass(frozen=True)
class DeepEmotionPrediction:
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    chunks_analyzed: int
    metadata: DeepEmotionMetadata


def normalize_label(label: str) -> str:
    raw = str(label).strip().lower().replace("_", " ").replace("-", " ")
    compact = " ".join(raw.split())
    return LABEL_ALIASES.get(compact, compact.replace(" ", "_"))


def normalize_probabilities(labels: list[str] | tuple[str, ...], values: np.ndarray) -> dict[str, float]:
    probabilities: dict[str, float] = {}
    for label, value in zip(labels, values, strict=True):
        probability = float(value)
        if not math.isfinite(probability) or probability < 0:
            raise DeepEmotionModelError("Deep emotion model returned invalid probabilities.")
        normalized = normalize_label(label)
        probabilities[normalized] = probabilities.get(normalized, 0.0) + probability

    total = sum(probabilities.values())
    if total <= 0 or not math.isfinite(total):
        raise DeepEmotionModelError("Deep emotion model returned an empty probability distribution.")

    return {label: value / total for label, value in probabilities.items()}


def chunk_audio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    max_seconds: float = 10.0,
    chunk_seconds: float = 8.0,
    overlap_seconds: float = 1.0,
) -> list[np.ndarray]:
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise DeepEmotionModelError("Cannot infer emotion from empty audio.")
    if sample_rate <= 0:
        raise DeepEmotionModelError("Sample rate must be positive.")
    if max_seconds <= 0 or chunk_seconds <= 0 or overlap_seconds < 0:
        raise DeepEmotionModelError("Deep-model chunk configuration is invalid.")
    if chunk_seconds > max_seconds:
        raise DeepEmotionModelError("Chunk duration cannot exceed the model maximum duration.")
    if overlap_seconds >= chunk_seconds:
        raise DeepEmotionModelError("Chunk overlap must be smaller than chunk duration.")

    max_samples = max(1, int(round(max_seconds * sample_rate)))
    chunk_samples = min(max_samples, max(1, int(round(chunk_seconds * sample_rate))))
    hop_samples = max(1, int(round((chunk_seconds - overlap_seconds) * sample_rate)))

    if array.size <= max_samples:
        return [array]

    chunks: list[np.ndarray] = []
    start = 0
    while start < array.size:
        end = min(start + chunk_samples, array.size)
        if end > start:
            chunks.append(array[start:end])
        if end >= array.size:
            break
        start += hop_samples
    return chunks


class DeepSpeechEmotionModel:
    """Lazy-loaded fine-tuned speech emotion classifier with long-audio chunking."""

    def __init__(
        self,
        *,
        model_id: str,
        processor: Any,
        model: Any,
        device: str,
        sample_rate: int = 16_000,
        max_seconds: float = 10.0,
        chunk_seconds: float = 8.0,
        overlap_seconds: float = 1.0,
    ) -> None:
        self.model_id = model_id
        self.processor = processor
        self.model = model
        self.device = device
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds

        config_labels = getattr(getattr(model, "config", None), "id2label", {}) or {}
        try:
            ordered_labels = [
                config_labels[index]
                for index in sorted(config_labels, key=lambda item: int(item))
            ]
        except (TypeError, ValueError):
            ordered_labels = list(config_labels.values())
        labels = [normalize_label(label) for label in ordered_labels] or list(STANDARD_LABELS)
        self.labels = tuple(dict.fromkeys(labels))

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        device: str = "auto",
        sample_rate: int = 16_000,
        max_seconds: float = 10.0,
        chunk_seconds: float = 8.0,
        overlap_seconds: float = 1.0,
    ) -> "DeepSpeechEmotionModel":
        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except Exception as exc:
            raise DeepEmotionModelError(
                "Deep speech inference requires the PyTorch and Transformers packages."
            ) from exc

        try:
            processor_cls = getattr(transformers, "AutoProcessor")
            model_cls = getattr(transformers, "AutoModelForAudioClassification")
        except AttributeError as exc:
            raise DeepEmotionModelError(
                "The installed Transformers version does not provide the required audio-classification classes."
            ) from exc

        resolved_device = cls._resolve_device(torch, device)
        try:
            processor = processor_cls.from_pretrained(model_id)
            model = model_cls.from_pretrained(model_id)
            model.to(resolved_device)
            model.eval()
        except Exception as exc:
            raise DeepEmotionModelError(
                f"Could not load deep emotion model '{model_id}'. The first load may download model files from Hugging Face."
            ) from exc

        return cls(
            model_id=model_id,
            processor=processor,
            model=model,
            device=str(resolved_device),
            sample_rate=sample_rate,
            max_seconds=max_seconds,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> Any:
        normalized = str(requested).strip().lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized == "cuda" and not torch.cuda.is_available():
            raise DeepEmotionModelError("CUDA was requested for deep emotion inference but is not available.")
        return torch.device(normalized)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "labels": list(self.labels),
            "sample_rate": self.sample_rate,
            "device": self.device,
            "max_seconds_per_chunk": self.max_seconds,
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
        }

    def _predict_chunk(self, chunk: np.ndarray) -> dict[str, float]:
        torch = importlib.import_module("torch")
        try:
            inputs = self.processor(
                chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
        except Exception as exc:
            raise DeepEmotionModelError("Deep emotion processor could not encode the audio chunk.") from exc

        moved_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        try:
            with torch.inference_mode():
                output = self.model(**moved_inputs)
                logits = output.logits
                probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()
        except Exception as exc:
            raise DeepEmotionModelError("Deep emotion model inference failed on an audio chunk.") from exc

        raw_config = getattr(getattr(self.model, "config", None), "id2label", {}) or {}
        try:
            labels = [
                str(raw_config[index])
                for index in sorted(raw_config, key=lambda item: int(item))
            ]
        except (TypeError, ValueError):
            labels = [str(label) for label in raw_config.values()]
        if not labels:
            labels = list(self.labels)
        if len(labels) != len(probabilities):
            raise DeepEmotionModelError(
                f"Model returned {len(probabilities)} scores but exposes {len(labels)} labels."
            )

        return normalize_probabilities(labels, probabilities)

    def predict(self, samples: np.ndarray, sample_rate: int) -> DeepEmotionPrediction:
        if int(sample_rate) != self.sample_rate:
            raise DeepEmotionModelError(
                f"Deep emotion model expects {self.sample_rate} Hz audio after preprocessing."
            )

        chunks = chunk_audio(
            samples,
            sample_rate,
            max_seconds=self.max_seconds,
            chunk_seconds=self.chunk_seconds,
            overlap_seconds=self.overlap_seconds,
        )
        chunk_predictions = [self._predict_chunk(chunk) for chunk in chunks]
        labels = sorted({label for prediction in chunk_predictions for label in prediction})
        aggregate = {
            label: float(np.mean([prediction.get(label, 0.0) for prediction in chunk_predictions]))
            for label in labels
        }
        total = sum(aggregate.values())
        if total <= 0 or not math.isfinite(total):
            raise DeepEmotionModelError("Deep emotion aggregation produced an invalid probability distribution.")
        aggregate = {label: value / total for label, value in aggregate.items()}

        emotion = max(aggregate, key=aggregate.get)
        confidence = float(aggregate[emotion])
        metadata = DeepEmotionMetadata(
            model_id=self.model_id,
            labels=tuple(labels),
            sample_rate=self.sample_rate,
            device=self.device,
        )
        return DeepEmotionPrediction(
            emotion=emotion,
            confidence=confidence,
            probabilities=aggregate,
            chunks_analyzed=len(chunks),
            metadata=metadata,
        )
