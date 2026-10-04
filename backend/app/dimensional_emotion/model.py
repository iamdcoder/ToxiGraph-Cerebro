from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


STANDARD_DIMENSIONS = ("valence", "arousal", "dominance")


class DimensionalEmotionError(RuntimeError):
    """Raised when dimensional speech-affect inference cannot be performed."""


@dataclass(frozen=True)
class DimensionalEmotionMetadata:
    model_id: str
    dimensions: tuple[str, ...]
    sample_rate: int
    device: str
    output_range: str


@dataclass(frozen=True)
class DimensionalEmotionPrediction:
    valence: float
    arousal: float
    dominance: float
    metadata: DimensionalEmotionMetadata

    @property
    def values(self) -> dict[str, float]:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
        }


def _finite_unit(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise DimensionalEmotionError(f"Dimensional model returned non-finite {name}.")
    if number < -1e-5 or number > 1.00001:
        raise DimensionalEmotionError(
            f"Dimensional model returned {name} outside the expected [0, 1] range."
        )
    return float(min(1.0, max(0.0, number)))


def _canonical_dimension(label: str) -> str:
    raw = str(label).strip().lower().replace("_", " ").replace("-", " ")
    compact = " ".join(raw.split())
    aliases = {
        "valence": "valence",
        "v": "valence",
        "arousal": "arousal",
        "a": "arousal",
        "dominance": "dominance",
        "dominance level": "dominance",
        "d": "dominance",
    }
    return aliases.get(compact, compact.replace(" ", "_"))


def parse_dimensional_output(labels: list[str], values: np.ndarray) -> dict[str, float]:
    raw_values = np.asarray(values, dtype=np.float32).reshape(-1)
    if raw_values.size != 3:
        raise DimensionalEmotionError(
            f"Dimensional model must return exactly 3 values; received {raw_values.size}."
        )

    normalized_labels = [_canonical_dimension(label) for label in labels]
    if set(normalized_labels) >= set(STANDARD_DIMENSIONS):
        by_dimension = dict(zip(normalized_labels, raw_values, strict=True))
        return {
            dimension: _finite_unit(by_dimension[dimension], dimension)
            for dimension in STANDARD_DIMENSIONS
        }

    # The configured reference model exposes the stable order Arousal, Dominance, Valence.
    return {
        "arousal": _finite_unit(raw_values[0], "arousal"),
        "dominance": _finite_unit(raw_values[1], "dominance"),
        "valence": _finite_unit(raw_values[2], "valence"),
    }


class DimensionalSpeechEmotionModel:
    """Lazy-loaded speech model for continuous Valence/Arousal/Dominance estimation."""

    def __init__(
        self,
        *,
        model_id: str,
        model: Any,
        processor: Any | None,
        device: str,
        sample_rate: int = 16_000,
        max_seconds: float = 30.0,
        chunk_seconds: float = 15.0,
        overlap_seconds: float = 3.0,
    ) -> None:
        self.model_id = model_id
        self.model = model
        self.processor = processor
        self.device = device
        self.sample_rate = sample_rate
        self.max_seconds = max_seconds
        self.chunk_seconds = chunk_seconds
        self.overlap_seconds = overlap_seconds

        config = getattr(model, "config", None)
        self.mean = getattr(config, "mean", None)
        self.std = getattr(config, "std", None)
        self.labels = self._read_labels(config)

    @staticmethod
    def _read_labels(config: Any) -> tuple[str, ...]:
        id2label = getattr(config, "id2label", {}) or {}
        ordered: list[str] = []
        try:
            ordered = [
                str(id2label[index])
                for index in sorted(id2label, key=lambda item: int(item))
            ]
        except (TypeError, ValueError):
            ordered = [str(value) for value in id2label.values()]
        return tuple(ordered or ("arousal", "dominance", "valence"))

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        device: str = "auto",
        sample_rate: int = 16_000,
        max_seconds: float = 30.0,
        chunk_seconds: float = 15.0,
        overlap_seconds: float = 3.0,
    ) -> "DimensionalSpeechEmotionModel":
        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except Exception as exc:
            raise DimensionalEmotionError(
                "Dimensional affect inference requires the PyTorch and Transformers packages."
            ) from exc

        try:
            model_cls = getattr(transformers, "AutoModelForAudioClassification")
            processor_cls = getattr(transformers, "AutoProcessor", None)
        except AttributeError as exc:
            raise DimensionalEmotionError(
                "The installed Transformers version does not provide the required audio model classes."
            ) from exc

        normalized_device = str(device).strip().lower()
        if normalized_device == "auto":
            resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif normalized_device == "cuda" and not torch.cuda.is_available():
            raise DimensionalEmotionError("CUDA was requested but is not available.")
        else:
            resolved_device = torch.device(normalized_device)

        try:
            model = model_cls.from_pretrained(model_id, trust_remote_code=True)
            model.to(resolved_device)
            model.eval()
            processor = None
        except Exception as exc:
            raise DimensionalEmotionError(
                f"Could not load dimensional affect model '{model_id}'. The first load may download model files from Hugging Face."
            ) from exc

        return cls(
            model_id=model_id,
            model=model,
            processor=processor,
            device=str(resolved_device),
            sample_rate=sample_rate,
            max_seconds=max_seconds,
            chunk_seconds=chunk_seconds,
            overlap_seconds=overlap_seconds,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "dimensions": list(STANDARD_DIMENSIONS),
            "native_labels": list(self.labels),
            "sample_rate": self.sample_rate,
            "device": self.device,
            "output_range": "[0, 1]",
            "max_seconds_per_chunk": self.max_seconds,
            "chunk_seconds": self.chunk_seconds,
            "overlap_seconds": self.overlap_seconds,
        }

    def _prepare_waveform(self, samples: np.ndarray) -> np.ndarray:
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise DimensionalEmotionError("Cannot infer dimensional affect from empty audio.")
        if not np.all(np.isfinite(array)):
            raise DimensionalEmotionError("Audio waveform contains non-finite values.")
        if self.mean is not None and self.std is not None:
            std = float(self.std)
            if not math.isfinite(std) or abs(std) < 1e-8:
                raise DimensionalEmotionError("Dimensional model normalization parameters are invalid.")
            array = (array - float(self.mean)) / (std + 1e-6)
        return array.astype(np.float32, copy=False)

    def _chunk_audio(self, samples: np.ndarray) -> list[np.ndarray]:
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        max_samples = max(1, int(round(self.max_seconds * self.sample_rate)))
        chunk_samples = min(max_samples, max(1, int(round(self.chunk_seconds * self.sample_rate))))
        hop_samples = max(1, int(round((self.chunk_seconds - self.overlap_seconds) * self.sample_rate)))
        if self.chunk_seconds > self.max_seconds or self.overlap_seconds >= self.chunk_seconds:
            raise DimensionalEmotionError("Dimensional model chunk configuration is invalid.")
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

    def _predict_chunk(self, chunk: np.ndarray) -> dict[str, float]:
        torch = importlib.import_module("torch")
        waveform = self._prepare_waveform(chunk)
        tensor = torch.from_numpy(waveform).unsqueeze(0).to(self.device)
        mask = torch.ones_like(tensor, dtype=torch.long)
        try:
            with torch.inference_mode():
                try:
                    output = self.model(tensor, mask)
                except TypeError:
                    output = self.model(input_values=tensor, attention_mask=mask)
        except Exception as exc:
            raise DimensionalEmotionError("Dimensional affect model inference failed.") from exc

        if hasattr(output, "logits"):
            values = output.logits
        elif isinstance(output, (tuple, list)):
            values = output[-1]
        else:
            values = output

        try:
            array = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
        except Exception as exc:
            raise DimensionalEmotionError("Dimensional affect model returned an unreadable output tensor.") from exc
        if array.ndim == 2:
            array = array[0]
        return parse_dimensional_output(list(self.labels), array)

    def predict(self, samples: np.ndarray, sample_rate: int) -> DimensionalEmotionPrediction:
        if sample_rate != self.sample_rate:
            raise DimensionalEmotionError(
                f"Dimensional model expects {self.sample_rate} Hz audio; received {sample_rate} Hz."
            )

        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise DimensionalEmotionError("Cannot infer dimensional affect from empty audio.")
        if not np.all(np.isfinite(array)):
            raise DimensionalEmotionError("Audio waveform contains non-finite values.")

        chunks = self._chunk_audio(array)
        chunk_results = [self._predict_chunk(chunk) for chunk in chunks]
        durations = np.asarray([max(1, len(chunk)) for chunk in chunks], dtype=np.float64)
        weights = durations / durations.sum()
        values = {
            dimension: float(sum(result[dimension] * weight for result, weight in zip(chunk_results, weights, strict=True)))
            for dimension in STANDARD_DIMENSIONS
        }

        metadata = DimensionalEmotionMetadata(
            model_id=self.model_id,
            dimensions=STANDARD_DIMENSIONS,
            sample_rate=self.sample_rate,
            device=self.device,
            output_range="[0, 1]",
        )
        return DimensionalEmotionPrediction(
            valence=values["valence"],
            arousal=values["arousal"],
            dominance=values["dominance"],
            metadata=metadata,
        )
