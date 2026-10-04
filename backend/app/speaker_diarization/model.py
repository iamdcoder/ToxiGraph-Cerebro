from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np


class SpeakerDiarizationError(RuntimeError):
    """Raised when speaker diarization cannot be performed."""


@dataclass(frozen=True)
class SpeakerTurn:
    speaker: str
    start_seconds: float
    end_seconds: float
    duration_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start_seconds": round(self.start_seconds, 4),
            "end_seconds": round(self.end_seconds, 4),
            "duration_seconds": round(self.duration_seconds, 4),
        }


class SpeakerDiarizationModel:
    """Lazy wrapper around the open-source pyannote Community-1 diarization pipeline."""

    def __init__(
        self,
        *,
        pipeline: Any,
        model_id: str,
        device: str,
        sample_rate: int = 16_000,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.model_id = model_id
        self.device = device
        self.sample_rate = sample_rate
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    @classmethod
    def load(
        cls,
        model_id: str,
        *,
        token: str | None,
        device: str = "auto",
        sample_rate: int = 16_000,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> "SpeakerDiarizationModel":
        if not token:
            raise SpeakerDiarizationError(
                "Speaker diarization requires a Hugging Face access token. Set PYANNOTE_TOKEN after accepting the model conditions."
            )

        try:
            pyannote_audio = importlib.import_module("pyannote.audio")
            torch = importlib.import_module("torch")
        except Exception as exc:
            raise SpeakerDiarizationError(
                "Speaker diarization requires the pyannote.audio and PyTorch packages."
            ) from exc

        pipeline_cls = getattr(pyannote_audio, "Pipeline", None)
        if pipeline_cls is None:
            raise SpeakerDiarizationError(
                "The installed pyannote.audio package does not expose Pipeline."
            )

        normalized_device = str(device).strip().lower()
        if normalized_device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif normalized_device == "cuda" and not torch.cuda.is_available():
            raise SpeakerDiarizationError("CUDA was requested but is not available.")
        else:
            resolved_device = normalized_device

        try:
            pipeline = pipeline_cls.from_pretrained(model_id, token=token)
            if resolved_device == "cuda" and hasattr(pipeline, "to"):
                pipeline.to(torch.device("cuda"))
        except Exception as exc:
            raise SpeakerDiarizationError(
                f"Could not load speaker diarization pipeline '{model_id}'. The first load may download model files from Hugging Face."
            ) from exc

        return cls(
            pipeline=pipeline,
            model_id=model_id,
            device=resolved_device,
            sample_rate=sample_rate,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "input_format": "mono 16 kHz waveform",
        }

    def diarize(self, samples: np.ndarray, sample_rate: int) -> list[SpeakerTurn]:
        if sample_rate != self.sample_rate:
            raise SpeakerDiarizationError(
                f"Speaker diarization expects {self.sample_rate} Hz audio; received {sample_rate} Hz."
            )
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise SpeakerDiarizationError("Cannot diarize an empty audio signal.")
        if not np.all(np.isfinite(array)):
            raise SpeakerDiarizationError("Audio waveform contains non-finite values.")
        try:
            torch = importlib.import_module("torch")
            waveform = torch.from_numpy(array).unsqueeze(0)
            request: dict[str, Any] = {
                "waveform": waveform,
                "sample_rate": sample_rate,
            }
            kwargs: dict[str, Any] = {}
            if self.min_speakers is not None:
                kwargs["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                kwargs["max_speakers"] = self.max_speakers
            output = self.pipeline(request, **kwargs)
        except SpeakerDiarizationError:
            raise
        except Exception as exc:
            raise SpeakerDiarizationError("Speaker diarization inference failed.") from exc

        return parse_diarization_output(output)


def parse_diarization_output(output: Any) -> list[SpeakerTurn]:
    """Normalize pyannote Annotation/DiarizeOutput variants into typed speaker turns."""
    annotation = getattr(output, "speaker_diarization", None)
    if annotation is None:
        annotation = output
    itertracks = getattr(annotation, "itertracks", None)
    if not callable(itertracks):
        raise SpeakerDiarizationError("Diarization output does not expose itertracks().")

    turns: list[SpeakerTurn] = []
    try:
        iterator = itertracks(yield_label=True)
        for item in iterator:
            if len(item) != 3:
                raise SpeakerDiarizationError("Unexpected diarization track format.")
            segment, _, speaker = item
            start = float(segment.start)
            end = float(segment.end)
            duration = end - start
            if not math.isfinite(start) or not math.isfinite(end) or duration <= 0:
                continue
            label = str(speaker).strip()
            if not label:
                continue
            turns.append(
                SpeakerTurn(
                    speaker=label,
                    start_seconds=start,
                    end_seconds=end,
                    duration_seconds=duration,
                )
            )
    except SpeakerDiarizationError:
        raise
    except Exception as exc:
        raise SpeakerDiarizationError("Could not parse diarization tracks.") from exc

    turns.sort(key=lambda turn: (turn.start_seconds, turn.end_seconds, turn.speaker))
    if not turns:
        raise SpeakerDiarizationError("No speakers were detected in the recording.")
    return turns
