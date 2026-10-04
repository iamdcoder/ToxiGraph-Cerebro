from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np

from app.fusion.engine import FusionError
from app.quality.engine import AudioQualityAnalyzer, AudioQualityError


class LiveAnalysisError(ValueError):
    """Raised when live streaming analysis cannot continue safely."""


@dataclass(frozen=True)
class LiveConfig:
    target_sample_rate: int = 16_000
    window_seconds: float = 3.0
    hop_seconds: float = 1.0
    min_window_seconds: float = 1.5
    max_duration_seconds: float = 120.0
    max_chunk_bytes: int = 256 * 1024
    min_speech_ratio: float = 0.30
    smoothing_alpha: float = 0.35


@dataclass(frozen=True)
class LiveInferenceResult:
    sequence: int
    window_start_seconds: float
    window_end_seconds: float
    emotion: str | None
    confidence: float
    probabilities: dict[str, float]
    valence: float | None
    arousal: float | None
    dominance: float | None
    state: str
    state_transition: str | None
    speech_ratio: float
    quality_score: float
    quality_verdict: str
    latency_ms: float
    model_metadata: dict[str, object]

    def to_dict(self) -> dict:
        return {
            "type": "analysis",
            "sequence": self.sequence,
            "window_start_seconds": round(self.window_start_seconds, 4),
            "window_end_seconds": round(self.window_end_seconds, 4),
            "emotion": self.emotion,
            "confidence": round(self.confidence, 6),
            "probabilities": {
                label: round(float(value), 6)
                for label, value in self.probabilities.items()
            },
            "valence": None if self.valence is None else round(self.valence, 6),
            "arousal": None if self.arousal is None else round(self.arousal, 6),
            "dominance": None if self.dominance is None else round(self.dominance, 6),
            "state": self.state,
            "state_transition": self.state_transition,
            "speech_ratio": round(self.speech_ratio, 6),
            "quality_score": round(self.quality_score, 6),
            "quality_verdict": self.quality_verdict,
            "latency_ms": round(self.latency_ms, 2),
            "model_metadata": self.model_metadata,
        }


@dataclass
class LiveStateMachine:
    current_state: str = "insufficient_data"
    previous_arousal: float | None = None
    previous_valence: float | None = None

    _high_arousal_emotions: frozenset[str] = field(
        default_factory=lambda: frozenset({"angry", "fear", "surprised", "disgust"})
    )
    _positive_emotions: frozenset[str] = field(
        default_factory=lambda: frozenset({"happy", "surprised"})
    )
    _negative_emotions: frozenset[str] = field(
        default_factory=lambda: frozenset({"angry", "fear", "sad", "disgust"})
    )

    def update(
        self,
        *,
        emotion: str | None,
        valence: float | None,
        arousal: float | None,
        speech_active: bool,
    ) -> tuple[str, str | None]:
        if not speech_active:
            return self.current_state if self.current_state != "insufficient_data" else "quiet", None

        if emotion is None and arousal is None and valence is None:
            return self.current_state, None

        previous = self.current_state
        if arousal is not None and valence is not None:
            state = self._state_from_dimensions(valence, arousal)
        else:
            state = self._state_from_emotion(emotion)

        transition = None if state == previous or previous == "insufficient_data" else f"{previous}_to_{state}"
        self.current_state = state
        self.previous_arousal = arousal
        self.previous_valence = valence
        return state, transition

    def _state_from_dimensions(self, valence: float, arousal: float) -> str:
        if (
            self.previous_arousal is not None
            and self.previous_arousal >= 0.62
            and arousal <= self.previous_arousal - 0.12
        ):
            return "cooling"
        if arousal >= 0.80 and valence <= 0.35:
            return "high_tension"
        if arousal >= 0.64 and valence < 0.50:
            return "tense"
        if valence >= 0.65 and arousal >= 0.40:
            return "engaged"
        if arousal <= 0.33:
            return "calm"
        if valence <= 0.40:
            return "tense"
        return "stable"

    def _state_from_emotion(self, emotion: str | None) -> str:
        normalized = (emotion or "").strip().lower()
        if normalized in self._high_arousal_emotions:
            return "activated"
        if normalized in self._positive_emotions:
            return "engaged"
        if normalized in self._negative_emotions:
            return "tense"
        if normalized == "neutral":
            return "calm"
        return "stable"


class LiveStreamSession:
    """Accumulate normalized audio and schedule one rolling window per hop."""

    def __init__(self, source_sample_rate: int, config: LiveConfig | None = None) -> None:
        if source_sample_rate < 8_000 or source_sample_rate > 96_000:
            raise LiveAnalysisError("Source sample rate must be between 8 kHz and 96 kHz.")
        self.config = config or LiveConfig()
        self._validate_config(self.config)
        self.source_sample_rate = int(source_sample_rate)
        self.samples = np.zeros(0, dtype=np.float32)
        self.sequence = 0
        self.last_scheduled_end = 0
        self.state_machine = LiveStateMachine()

    @staticmethod
    def _validate_config(config: LiveConfig) -> None:
        if config.target_sample_rate <= 0:
            raise LiveAnalysisError("Target sample rate must be positive.")
        if config.window_seconds <= 0:
            raise LiveAnalysisError("Live window duration must be positive.")
        if config.hop_seconds <= 0 or config.hop_seconds > config.window_seconds:
            raise LiveAnalysisError("Live hop must be positive and cannot exceed the window duration.")
        if config.min_window_seconds <= 0 or config.min_window_seconds > config.window_seconds:
            raise LiveAnalysisError("Live minimum window duration is invalid.")
        if config.max_duration_seconds <= config.window_seconds:
            raise LiveAnalysisError("Live maximum duration must exceed one full analysis window.")
        if config.max_chunk_bytes <= 0:
            raise LiveAnalysisError("Live chunk size must be positive.")
        if not 0 < config.min_speech_ratio <= 1:
            raise LiveAnalysisError("Minimum live speech ratio must be between 0 and 1.")
        if not 0 < config.smoothing_alpha <= 1:
            raise LiveAnalysisError("Live smoothing alpha must be between 0 and 1.")

    @property
    def duration_seconds(self) -> float:
        return float(len(self.samples) / self.config.target_sample_rate)

    def append_pcm16(self, chunk: bytes) -> float:
        if not chunk:
            raise LiveAnalysisError("Live audio chunk is empty.")
        if len(chunk) > self.config.max_chunk_bytes:
            raise LiveAnalysisError("Live audio chunk is too large.")
        if len(chunk) % 2:
            raise LiveAnalysisError("Live PCM16 chunk has an incomplete sample.")

        pcm = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0
        normalized = self._resample(
            pcm,
            source_rate=self.source_sample_rate,
            target_rate=self.config.target_sample_rate,
        )
        self.samples = np.concatenate((self.samples, normalized.astype(np.float32, copy=False)))
        if self.duration_seconds > self.config.max_duration_seconds + 1e-9:
            raise LiveAnalysisError(
                f"Live recording exceeded the {self.config.max_duration_seconds:.0f}-second limit."
            )
        return self.duration_seconds

    def append_float32(self, samples: np.ndarray) -> float:
        array = np.asarray(samples, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise LiveAnalysisError("Live audio chunk is empty.")
        if not np.isfinite(array).all():
            raise LiveAnalysisError("Live audio chunk contains non-finite samples.")
        pcm = np.clip(array, -1.0, 1.0)
        int16 = (pcm * 32767.0).astype("<i2")
        return self.append_pcm16(int16.tobytes())

    def pending_window(self, *, final: bool = False) -> tuple[np.ndarray, float, float] | None:
        available = self.duration_seconds
        window_end = available
        window = self.config.window_seconds
        if available >= window - 1e-6:
            elapsed_since_schedule = available - (self.last_scheduled_end / self.config.target_sample_rate)
            if elapsed_since_schedule < self.config.hop_seconds - 1e-6:
                return None
            start = max(0.0, window_end - window)
        elif final and available >= self.config.min_window_seconds:
            if available - (self.last_scheduled_end / self.config.target_sample_rate) < self.config.min_window_seconds - 1e-6:
                return None
            start = 0.0
        else:
            return None

        start_index = int(round(start * self.config.target_sample_rate))
        end_index = len(self.samples)
        snapshot = self.samples[start_index:end_index].copy()
        if snapshot.size == 0:
            return None
        self.last_scheduled_end = end_index
        return snapshot, start, window_end

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    @staticmethod
    def _resample(samples: np.ndarray, *, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return samples.astype(np.float32, copy=True)
        if samples.size == 0:
            return samples.astype(np.float32)
        source_length = len(samples)
        target_length = max(1, int(round(source_length * target_rate / source_rate)))
        source_positions = np.arange(source_length, dtype=np.float64)
        target_positions = np.linspace(0.0, max(0.0, source_length - 1), target_length, dtype=np.float64)
        return np.interp(target_positions, source_positions, samples.astype(np.float64)).astype(np.float32)


class LiveInferenceEngine:
    """Run one rolling live window through CEREBRO's existing inference stack."""

    def __init__(
        self,
        *,
        predictor: Callable[[np.ndarray, int], Mapping[str, object]],
        config: LiveConfig | None = None,
        quality_analyzer: AudioQualityAnalyzer | None = None,
    ) -> None:
        self.predictor = predictor
        self.config = config or LiveConfig()
        LiveStreamSession._validate_config(self.config)
        self.quality = quality_analyzer or AudioQualityAnalyzer()

    def analyze_window(
        self,
        session: LiveStreamSession,
        samples: np.ndarray,
        start_seconds: float,
        end_seconds: float,
        *,
        latency_ms: float,
    ) -> LiveInferenceResult:
        try:
            quality = self.quality.analyze(samples, self.config.target_sample_rate)
        except AudioQualityError as exc:
            raise LiveAnalysisError(str(exc)) from exc

        speech_active = quality.speech_ratio >= self.config.min_speech_ratio and not quality.rms <= 0.003
        if not speech_active:
            state, transition = session.state_machine.update(
                emotion=None,
                valence=None,
                arousal=None,
                speech_active=False,
            )
            return LiveInferenceResult(
                sequence=session.next_sequence(),
                window_start_seconds=start_seconds,
                window_end_seconds=end_seconds,
                emotion=None,
                confidence=0.0,
                probabilities={},
                valence=None,
                arousal=None,
                dominance=None,
                state=state if state != "insufficient_data" else "quiet",
                state_transition=transition,
                speech_ratio=quality.speech_ratio,
                quality_score=quality.quality_score,
                quality_verdict=quality.verdict,
                latency_ms=latency_ms,
                model_metadata={"mode": "quality_gate"},
            )

        try:
            output = dict(self.predictor(samples, self.config.target_sample_rate))
        except (FusionError, Exception) as exc:
            raise LiveAnalysisError(f"Live inference failed: {exc}") from exc

        probabilities = {
            str(label): float(value)
            for label, value in dict(output.get("probabilities", {})).items()
        }
        if probabilities:
            total = sum(max(0.0, value) for value in probabilities.values())
            if total <= 0 or not math.isfinite(total):
                raise LiveAnalysisError("Live inference returned invalid probabilities.")
            probabilities = {
                label: max(0.0, value) / total
                for label, value in probabilities.items()
            }
        emotion = output.get("emotion")
        confidence = float(output.get("confidence", max(probabilities.values()) if probabilities else 0.0))
        valence = self._optional_float(output.get("valence"))
        arousal = self._optional_float(output.get("arousal"))
        dominance = self._optional_float(output.get("dominance"))
        state, transition = session.state_machine.update(
            emotion=str(emotion) if emotion is not None else None,
            valence=valence,
            arousal=arousal,
            speech_active=True,
        )
        return LiveInferenceResult(
            sequence=session.next_sequence(),
            window_start_seconds=start_seconds,
            window_end_seconds=end_seconds,
            emotion=str(emotion) if emotion is not None else None,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            probabilities=probabilities,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            state=state,
            state_transition=transition,
            speech_ratio=quality.speech_ratio,
            quality_score=quality.quality_score,
            quality_verdict=quality.verdict,
            latency_ms=latency_ms,
            model_metadata=dict(output.get("model_metadata", {})),
        )

    @staticmethod
    def _optional_float(value) -> float | None:
        if value is None:
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        return float(np.clip(number, 0.0, 1.0))
