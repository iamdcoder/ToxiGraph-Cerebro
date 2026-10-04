from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass

import numpy as np


class AudioProcessingError(ValueError):
    """Raised when an uploaded audio file cannot be processed safely."""


@dataclass(frozen=True)
class AudioConfig:
    target_sample_rate: int = 16_000
    min_duration_seconds: float = 0.5
    max_duration_seconds: float = 60.0
    max_bytes: int = 12 * 1024 * 1024
    silence_rms_threshold: float = 0.003


@dataclass(frozen=True)
class ProcessedAudio:
    samples: np.ndarray
    sample_rate: int
    channels: int
    original_sample_rate: int
    original_channels: int
    original_sample_width_bytes: int
    duration_seconds: float
    rms: float
    peak: float
    is_silent: bool

    def to_metadata(self) -> dict:
        return {
            "format": "wav",
            "encoding": "pcm",
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "original_sample_rate": self.original_sample_rate,
            "original_channels": self.original_channels,
            "original_sample_width_bytes": self.original_sample_width_bytes,
            "duration_seconds": round(self.duration_seconds, 4),
            "rms": round(self.rms, 6),
            "peak": round(self.peak, 6),
            "is_silent": self.is_silent,
        }


class AudioProcessor:
    """Validate PCM WAV input and normalize it to mono float32 at 16 kHz."""

    def __init__(self, config: AudioConfig | None = None) -> None:
        self.config = config or AudioConfig()

    def process(self, audio_bytes: bytes) -> ProcessedAudio:
        self._validate_size(audio_bytes)

        try:
            with wave.open(io.BytesIO(audio_bytes), "rb") as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                sample_width = wav.getsampwidth()
                frame_count = wav.getnframes()
                compression = wav.getcomptype()
                raw_frames = wav.readframes(frame_count)
        except (wave.Error, EOFError, ValueError) as exc:
            raise AudioProcessingError(
                "Invalid WAV audio. Upload a PCM WAV recording."
            ) from exc

        if compression != "NONE":
            raise AudioProcessingError("Compressed WAV audio is not supported.")

        self._validate_header(
            channels=channels,
            sample_rate=sample_rate,
            sample_width=sample_width,
            frame_count=frame_count,
        )

        samples = self._decode_pcm(
            raw_frames,
            channels=channels,
            sample_width=sample_width,
        )

        mono = self._to_mono(samples, channels)
        normalized = mono.astype(np.float32, copy=False)
        resampled = self._resample(
            normalized,
            source_rate=sample_rate,
            target_rate=self.config.target_sample_rate,
        )

        duration = len(resampled) / self.config.target_sample_rate
        self._validate_duration(duration)

        rms = float(np.sqrt(np.mean(np.square(resampled), dtype=np.float64)))
        peak = float(np.max(np.abs(resampled))) if resampled.size else 0.0
        is_silent = rms < self.config.silence_rms_threshold

        return ProcessedAudio(
            samples=resampled,
            sample_rate=self.config.target_sample_rate,
            channels=1,
            original_sample_rate=sample_rate,
            original_channels=channels,
            original_sample_width_bytes=sample_width,
            duration_seconds=duration,
            rms=rms,
            peak=peak,
            is_silent=is_silent,
        )

    def _validate_size(self, audio_bytes: bytes) -> None:
        size = len(audio_bytes)
        if size == 0:
            raise AudioProcessingError("The uploaded audio file is empty.")
        if size > self.config.max_bytes:
            max_mb = self.config.max_bytes / (1024 * 1024)
            raise AudioProcessingError(
                f"Audio file is too large. Maximum size is {max_mb:.0f} MB."
            )

    def _validate_header(
        self,
        *,
        channels: int,
        sample_rate: int,
        sample_width: int,
        frame_count: int,
    ) -> None:
        if channels not in (1, 2):
            raise AudioProcessingError("Only mono or stereo WAV audio is supported.")
        if sample_rate < 8_000 or sample_rate > 96_000:
            raise AudioProcessingError(
                "Sample rate must be between 8 kHz and 96 kHz."
            )
        if sample_width not in (1, 2, 3, 4):
            raise AudioProcessingError(
                "Only 8-bit, 16-bit, 24-bit, and 32-bit PCM WAV audio is supported."
            )
        if frame_count <= 0:
            raise AudioProcessingError("The WAV file contains no audio frames.")

    def _validate_duration(self, duration: float) -> None:
        if duration < self.config.min_duration_seconds:
            raise AudioProcessingError(
                f"Audio is too short. Minimum duration is "
                f"{self.config.min_duration_seconds:.1f} seconds."
            )
        if duration > self.config.max_duration_seconds:
            raise AudioProcessingError(
                f"Audio is too long. Maximum duration is "
                f"{self.config.max_duration_seconds:.0f} seconds."
            )

    @staticmethod
    def _to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
        if channels == 1:
            return samples
        return np.mean(samples, axis=1, dtype=np.float32)

    @staticmethod
    def _decode_pcm(
        raw_frames: bytes,
        *,
        channels: int,
        sample_width: int,
    ) -> np.ndarray:
        if sample_width == 1:
            data = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sample_width == 2:
            data = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32)
            data /= 32768.0
        elif sample_width == 3:
            raw = np.frombuffer(raw_frames, dtype=np.uint8).reshape(-1, 3)
            unsigned = (
                raw[:, 0].astype(np.int32)
                | (raw[:, 1].astype(np.int32) << 8)
                | (raw[:, 2].astype(np.int32) << 16)
            )
            signed = np.where(
                unsigned & 0x800000,
                unsigned - 0x1000000,
                unsigned,
            ).astype(np.float32)
            data = signed / 8388608.0
        else:
            data = np.frombuffer(raw_frames, dtype="<i4").astype(np.float32)
            data /= 2147483648.0

        if data.size % channels != 0:
            raise AudioProcessingError("WAV frame data is malformed.")

        return data.reshape(-1, channels)

    @staticmethod
    def _resample(
        samples: np.ndarray,
        *,
        source_rate: int,
        target_rate: int,
    ) -> np.ndarray:
        if source_rate == target_rate:
            return samples.astype(np.float32, copy=True)

        if samples.size == 0:
            return samples.astype(np.float32)

        source_length = len(samples)
        target_length = max(
            1,
            int(round(source_length * target_rate / source_rate)),
        )

        source_positions = np.arange(source_length, dtype=np.float64)
        target_positions = np.linspace(
            0.0,
            max(0.0, source_length - 1),
            target_length,
            dtype=np.float64,
        )

        resampled = np.interp(
            target_positions,
            source_positions,
            samples.astype(np.float64),
        )
        return resampled.astype(np.float32)
