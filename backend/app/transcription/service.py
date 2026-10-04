from __future__ import annotations

import importlib
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np


class ASRTranscriptionError(RuntimeError):
    """Raised when local automatic speech recognition cannot be completed."""


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    model_id: str
    language: str | None
    duration_seconds: float
    model_loaded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model_id": self.model_id,
            "language": self.language,
            "duration_seconds": round(self.duration_seconds, 4),
            "model_loaded": self.model_loaded,
        }


class WhisperTranscriber:
    """Lazy-loaded local Whisper transcription service."""

    def __init__(
        self,
        *,
        model_id: str,
        processor: Any,
        model: Any,
        device: str,
        sample_rate: int = 16_000,
        language: str | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        self.model_id = model_id
        self.processor = processor
        self.model = model
        self.device = device
        self.sample_rate = sample_rate
        self.language = language
        self.max_new_tokens = max_new_tokens

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        device: str = "auto",
        sample_rate: int = 16_000,
        language: str | None = None,
        max_new_tokens: int = 128,
    ) -> "WhisperTranscriber":
        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except Exception as exc:
            raise ASRTranscriptionError(
                "Speech transcription requires the PyTorch and Transformers packages."
            ) from exc

        try:
            processor_cls = getattr(transformers, "AutoProcessor")
            model_cls = getattr(transformers, "AutoModelForSpeechSeq2Seq")
        except AttributeError as exc:
            raise ASRTranscriptionError(
                "The installed Transformers version does not provide the required Whisper ASR classes."
            ) from exc

        resolved_device = cls._resolve_device(torch, device)
        try:
            processor = processor_cls.from_pretrained(model_id)
            model = model_cls.from_pretrained(model_id)
            model.to(resolved_device)
            model.eval()
        except Exception as exc:
            raise ASRTranscriptionError(
                f"Could not load ASR model '{model_id}'. The first load may download model files from Hugging Face."
            ) from exc

        return cls(
            model_id=model_id,
            processor=processor,
            model=model,
            device=str(resolved_device),
            sample_rate=sample_rate,
            language=language,
            max_new_tokens=max_new_tokens,
        )

    @staticmethod
    def _resolve_device(torch: Any, requested: str) -> Any:
        normalized = str(requested).strip().lower()
        if normalized == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if normalized == "cuda" and not torch.cuda.is_available():
            raise ASRTranscriptionError("CUDA was requested for ASR, but it is not available.")
        return torch.device(normalized)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "task": "automatic_speech_recognition",
            "device": self.device,
            "sample_rate": self.sample_rate,
            "language": self.language,
            "max_new_tokens": self.max_new_tokens,
        }

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> TranscriptionResult:
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise ASRTranscriptionError("Cannot transcribe empty audio.")
        if sample_rate != self.sample_rate:
            raise ASRTranscriptionError(
                f"ASR expects {self.sample_rate} Hz audio, received {sample_rate} Hz."
            )
        duration = float(array.size / sample_rate)
        if not np.isfinite(array).all():
            raise ASRTranscriptionError("Audio contains non-finite samples.")

        if float(np.sqrt(np.mean(np.square(array), dtype=np.float64))) < 0.001:
            return TranscriptionResult(
                text="",
                model_id=self.model_id,
                language=self.language,
                duration_seconds=duration,
            )

        torch = importlib.import_module("torch")
        try:
            inputs = self.processor(
                array,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
        except Exception as exc:
            raise ASRTranscriptionError("Whisper processor could not encode the audio.") from exc

        moved_inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        generation_kwargs: dict[str, Any] = {"max_new_tokens": self.max_new_tokens}
        if self.language:
            get_prompt = getattr(self.processor, "get_decoder_prompt_ids", None)
            if callable(get_prompt):
                try:
                    generation_kwargs["forced_decoder_ids"] = get_prompt(
                        language=self.language,
                        task="transcribe",
                    )
                except Exception:
                    pass

        try:
            with torch.inference_mode():
                generated = self.model.generate(**moved_inputs, **generation_kwargs)
                decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        except Exception as exc:
            raise ASRTranscriptionError("Whisper ASR inference failed.") from exc

        text = " ".join(str(decoded[0] if decoded else "").split()).strip()
        return TranscriptionResult(
            text=text,
            model_id=self.model_id,
            language=self.language,
            duration_seconds=duration,
        )


_transcriber: WhisperTranscriber | None = None
_transcriber_lock = Lock()


def get_transcriber() -> WhisperTranscriber:
    from app.config import settings

    global _transcriber
    if _transcriber is not None:
        return _transcriber
    with _transcriber_lock:
        if _transcriber is None:
            _transcriber = WhisperTranscriber.load(
                settings.ASR_MODEL,
                device=settings.ASR_DEVICE,
                sample_rate=settings.ASR_SAMPLE_RATE,
                language=settings.ASR_LANGUAGE,
                max_new_tokens=settings.ASR_MAX_NEW_TOKENS,
            )
    return _transcriber


def clear_transcriber_cache() -> None:
    global _transcriber
    with _transcriber_lock:
        _transcriber = None
