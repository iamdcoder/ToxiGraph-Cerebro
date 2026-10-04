from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class AudioQualityError(ValueError):
    """Raised when audio-quality analysis receives invalid data."""


@dataclass(frozen=True)
class AudioQualityReport:
    quality_score: float
    verdict: str
    recommendation: str
    duration_seconds: float
    rms: float
    peak: float
    speech_ratio: float
    clipping_ratio: float
    estimated_snr_db: float | None
    noise_floor_rms: float | None
    speech_rms: float
    dynamic_range_db: float
    dc_offset: float
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_score": round(self.quality_score, 4),
            "verdict": self.verdict,
            "recommendation": self.recommendation,
            "duration_seconds": round(self.duration_seconds, 4),
            "rms": round(self.rms, 6),
            "peak": round(self.peak, 6),
            "speech_ratio": round(self.speech_ratio, 4),
            "clipping_ratio": round(self.clipping_ratio, 6),
            "estimated_snr_db": None if self.estimated_snr_db is None else round(self.estimated_snr_db, 3),
            "noise_floor_rms": None if self.noise_floor_rms is None else round(self.noise_floor_rms, 6),
            "speech_rms": round(self.speech_rms, 6),
            "dynamic_range_db": round(self.dynamic_range_db, 3),
            "dc_offset": round(self.dc_offset, 6),
            "issues": list(self.issues),
        }


@dataclass(frozen=True)
class AudioQualityConfig:
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    speech_frame_threshold_ratio: float = 0.20
    minimum_snr_db: float = 6.0
    good_snr_db: float = 18.0
    clipping_warn_ratio: float = 0.002
    clipping_bad_ratio: float = 0.010
    excessive_silence_ratio: float = 0.55
    level_low_rms: float = 0.008
    level_high_rms: float = 0.40
    dc_offset_warn: float = 0.02


class AudioQualityAnalyzer:
    """Estimate whether an audio sample is technically suitable for emotion analysis."""

    def __init__(self, config: AudioQualityConfig | None = None) -> None:
        self.config = config or AudioQualityConfig()

    def analyze(
        self,
        samples: np.ndarray,
        sample_rate: int,
        *,
        speech_ratio: float | None = None,
    ) -> AudioQualityReport:
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise AudioQualityError("Audio quality analysis requires non-empty samples.")
        if sample_rate <= 0:
            raise AudioQualityError("Sample rate must be positive.")
        if not np.isfinite(array).all():
            raise AudioQualityError("Audio quality analysis received non-finite samples.")

        duration = float(array.size / sample_rate)
        rms = float(np.sqrt(np.mean(np.square(array), dtype=np.float64)))
        peak = float(np.max(np.abs(array)))
        clipping_ratio = float(np.mean(np.abs(array) >= 0.99))
        dc_offset = float(abs(np.mean(array)))
        dynamic_range_db = float(20.0 * math.log10(max(peak / max(rms, 1e-9), 1.0)))

        frame_length = max(1, int(round(sample_rate * self.config.frame_ms / 1000.0)))
        hop_length = max(1, int(round(sample_rate * self.config.hop_ms / 1000.0)))
        frames = self._frame_signal(array, frame_length, hop_length)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))

        if speech_ratio is None:
            speech_mask = self._speech_mask(frame_rms)
            measured_speech_ratio = float(np.mean(speech_mask)) if speech_mask.size else 0.0
        else:
            measured_speech_ratio = float(np.clip(speech_ratio, 0.0, 1.0))
            speech_mask = self._speech_mask(frame_rms)

        speech_frames = frame_rms[speech_mask]
        nonspeech_frames = frame_rms[~speech_mask]
        speech_rms = float(np.percentile(speech_frames, 75)) if speech_frames.size else float(rms)
        noise_floor = float(np.percentile(nonspeech_frames, 50)) if nonspeech_frames.size else None
        if noise_floor is not None and noise_floor > 1e-8 and speech_rms > 0:
            estimated_snr_db = float(20.0 * math.log10(speech_rms / noise_floor))
        else:
            estimated_snr_db = None

        issues: list[str] = []
        if duration < 1.0:
            issues.append("very_short_recording")
        elif duration < 2.0:
            issues.append("short_recording")
        if measured_speech_ratio < 0.35:
            issues.append("excessive_silence")
        if rms < self.config.level_low_rms:
            issues.append("low_recording_level")
        if rms > self.config.level_high_rms:
            issues.append("high_recording_level")
        if clipping_ratio >= self.config.clipping_bad_ratio:
            issues.append("clipping")
        elif clipping_ratio >= self.config.clipping_warn_ratio:
            issues.append("possible_clipping")
        if estimated_snr_db is not None and estimated_snr_db < self.config.minimum_snr_db:
            issues.append("low_signal_to_noise_ratio")
        if dc_offset >= self.config.dc_offset_warn:
            issues.append("dc_offset")

        duration_score = 1.0 if 1.5 <= duration <= 30.0 else 0.75 if duration >= 1.0 else 0.45
        silence_score = float(np.clip((measured_speech_ratio - 0.20) / 0.65, 0.0, 1.0))
        level_score = self._level_score(rms)
        clipping_score = float(np.clip(1.0 - clipping_ratio / max(self.config.clipping_bad_ratio, 1e-9), 0.0, 1.0))
        dc_score = float(np.clip(1.0 - dc_offset / max(self.config.dc_offset_warn * 2.0, 1e-9), 0.0, 1.0))
        if estimated_snr_db is None:
            snr_score = 0.65
        else:
            snr_score = float(np.clip((estimated_snr_db - 2.0) / (self.config.good_snr_db - 2.0), 0.0, 1.0))

        score = float(np.clip(
            0.25 * snr_score
            + 0.20 * silence_score
            + 0.20 * clipping_score
            + 0.15 * level_score
            + 0.10 * duration_score
            + 0.10 * dc_score,
            0.0,
            1.0,
        ))

        if "clipping" in issues or "low_recording_level" in issues or "high_recording_level" in issues or "excessive_silence" in issues or (estimated_snr_db is not None and estimated_snr_db < self.config.minimum_snr_db):
            verdict = "poor" if score >= 0.45 else "unreliable"
        elif score >= 0.80:
            verdict = "good"
        elif score >= 0.60:
            verdict = "acceptable"
        else:
            verdict = "poor"

        recommendation = self._recommendation(verdict, issues)
        return AudioQualityReport(
            quality_score=score,
            verdict=verdict,
            recommendation=recommendation,
            duration_seconds=duration,
            rms=rms,
            peak=peak,
            speech_ratio=measured_speech_ratio,
            clipping_ratio=clipping_ratio,
            estimated_snr_db=estimated_snr_db,
            noise_floor_rms=noise_floor,
            speech_rms=speech_rms,
            dynamic_range_db=dynamic_range_db,
            dc_offset=dc_offset,
            issues=issues,
        )

    def _speech_mask(self, frame_rms: np.ndarray) -> np.ndarray:
        if frame_rms.size == 0:
            return np.zeros(0, dtype=bool)
        if float(np.max(frame_rms)) <= 1e-8:
            return np.zeros(frame_rms.shape, dtype=bool)
        reference = float(np.percentile(frame_rms, 90))
        floor = float(np.percentile(frame_rms, 20))
        threshold = floor + self.config.speech_frame_threshold_ratio * max(reference - floor, 1e-9)
        if threshold <= 1e-6:
            threshold = reference * 0.15
        return frame_rms >= threshold

    @staticmethod
    def _frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
        frame_count = max(1, 1 + int(math.ceil(max(0, len(samples) - frame_length) / hop_length)))
        padded_length = (frame_count - 1) * hop_length + frame_length
        padded = np.pad(samples.astype(np.float64), (0, max(0, padded_length - len(samples))))
        starts = np.arange(frame_count) * hop_length
        return np.stack([padded[start:start + frame_length] for start in starts])

    def _level_score(self, rms: float) -> float:
        if rms <= 0:
            return 0.0
        if rms < self.config.level_low_rms:
            return float(np.clip(rms / self.config.level_low_rms, 0.0, 1.0))
        if rms <= 0.12:
            return 1.0
        return float(np.clip(1.0 - (rms - 0.12) / max(self.config.level_high_rms - 0.12, 1e-9), 0.0, 1.0))

    @staticmethod
    def _recommendation(verdict: str, issues: list[str]) -> str:
        if verdict == "good":
            return "Audio quality is suitable for interpretation."
        if not issues:
            return "Audio is usable, but confidence should be interpreted with normal caution."
        if "clipping" in issues:
            return "Reduce microphone input gain and record again to avoid clipping."
        if "low_recording_level" in issues:
            return "Move closer to the microphone or increase input gain slightly."
        if "excessive_silence" in issues:
            return "Record a clearer, more continuous voice sample."
        if "low_signal_to_noise_ratio" in issues:
            return "Move to a quieter environment or use a closer microphone."
        return "Record a clearer sample with steady microphone distance and less background noise."


def analyze_audio_quality(samples: np.ndarray, sample_rate: int, *, speech_ratio: float | None = None) -> AudioQualityReport:
    return AudioQualityAnalyzer().analyze(samples, sample_rate, speech_ratio=speech_ratio)
