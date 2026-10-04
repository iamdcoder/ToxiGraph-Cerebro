from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AcousticConfig:
    frame_length_ms: float = 25.0
    hop_length_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 26
    n_mfcc: int = 13
    fmin_hz: float = 50.0
    fmax_hz: float = 7600.0
    silence_multiplier: float = 0.30
    min_voiced_rms: float = 0.005


FEATURE_VECTOR_NAMES = [
    f"{metric}_{stat}"
    for metric in (
        "pitch_hz",
        "energy",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_hz",
        "spectral_flatness",
        "zero_crossing_rate",
    )
    for stat in ("mean", "std", "min", "max")
] + [
    f"spectral_contrast_db_{index}" for index in range(6)
] + [
    f"chroma_{index}" for index in range(12)
] + [
    f"mfcc_mean_{index}" for index in range(13)
] + [
    f"mfcc_std_{index}" for index in range(13)
] + [
    "duration_seconds",
    "silence_ratio",
    "speech_duration_seconds",
    "estimated_syllable_rate_sps",
    "estimated_pause_count",
]


@dataclass(frozen=True)
class AcousticFeatures:
    version: str
    sample_rate: int
    duration_seconds: float
    frame_count: int
    frame_length_seconds: float
    hop_length_seconds: float
    voiced_frame_count: int
    silence_ratio: float
    speech_duration_seconds: float
    estimated_syllable_rate_sps: float
    estimated_pause_count: int
    estimated_pause_ratio: float
    pitch_hz: dict[str, float]
    energy: dict[str, float]
    spectral_centroid_hz: dict[str, float]
    spectral_bandwidth_hz: dict[str, float]
    spectral_rolloff_hz: dict[str, float]
    spectral_flatness: dict[str, float]
    spectral_contrast_db: list[float]
    zero_crossing_rate: dict[str, float]
    chroma: list[float]
    mfcc_mean: list[float]
    mfcc_std: list[float]

    def to_vector(self) -> np.ndarray:
        return _feature_vector(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "sample_rate": self.sample_rate,
            "duration_seconds": round(self.duration_seconds, 4),
            "frame_count": self.frame_count,
            "frame_length_seconds": round(self.frame_length_seconds, 5),
            "hop_length_seconds": round(self.hop_length_seconds, 5),
            "voiced_frame_count": self.voiced_frame_count,
            "silence_ratio": round(self.silence_ratio, 4),
            "speech_duration_seconds": round(self.speech_duration_seconds, 4),
            "estimated_syllable_rate_sps": round(self.estimated_syllable_rate_sps, 3),
            "estimated_pause_count": self.estimated_pause_count,
            "estimated_pause_ratio": round(self.estimated_pause_ratio, 4),
            "pitch_hz": _rounded_stats(self.pitch_hz),
            "energy": _rounded_stats(self.energy),
            "spectral_centroid_hz": _rounded_stats(self.spectral_centroid_hz),
            "spectral_bandwidth_hz": _rounded_stats(self.spectral_bandwidth_hz),
            "spectral_rolloff_hz": _rounded_stats(self.spectral_rolloff_hz),
            "spectral_flatness": _rounded_stats(self.spectral_flatness),
            "spectral_contrast_db": [round(float(value), 5) for value in self.spectral_contrast_db],
            "zero_crossing_rate": _rounded_stats(self.zero_crossing_rate),
            "chroma": [round(float(value), 5) for value in self.chroma],
            "mfcc_mean": [round(float(value), 5) for value in self.mfcc_mean],
            "mfcc_std": [round(float(value), 5) for value in self.mfcc_std],
            "model_vector_length": int(self.to_vector().size),
        }


def _rounded_stats(values: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 5) for key, value in values.items()}


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else 0.0


def _safe_std(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else 0.0


def _safe_min(values: np.ndarray) -> float:
    return float(np.min(values)) if values.size else 0.0


def _safe_max(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else 0.0


def _stats(values: np.ndarray, *, positive_only: bool = False) -> dict[str, float]:
    clean = values[np.isfinite(values)]
    if positive_only:
        clean = clean[clean > 0]
    return {
        "mean": _safe_mean(clean),
        "std": _safe_std(clean),
        "min": _safe_min(clean),
        "max": _safe_max(clean),
    }


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    fmax = min(fmax, sample_rate / 2.0)
    if fmax <= fmin:
        raise ValueError("fmax must be greater than fmin for the mel filterbank.")

    mel_points = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz_points = np.array([_mel_to_hz(value) for value in mel_points])
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filters = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for index in range(n_mels):
        left, center, right = int(bins[index]), int(bins[index + 1]), int(bins[index + 2])
        if center <= left:
            center = min(left + 1, n_fft // 2)
        if right <= center:
            right = min(center + 1, n_fft // 2 + 1)

        if center > left:
            filters[index, left:center] = (
                np.arange(left, center) - left
            ) / (center - left)
        if right > center:
            filters[index, center:right] = (
                right - np.arange(center, right)
            ) / (right - center)

    row_sums = filters.sum(axis=1, keepdims=True)
    filters /= np.maximum(row_sums, 1e-12)
    return filters


def _dct_type_ii(values: np.ndarray, count: int) -> np.ndarray:
    n = values.shape[-1]
    index = np.arange(n, dtype=np.float64)
    basis_index = np.arange(count, dtype=np.float64)[:, None]
    basis = np.cos(math.pi / n * (index + 0.5) * basis_index)
    basis[0] *= 1.0 / math.sqrt(n)
    if count > 1:
        basis[1:] *= math.sqrt(2.0 / n)
    return values @ basis.T


def _frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if samples.size == 0:
        return np.zeros((1, frame_length), dtype=np.float64)

    frame_length = max(frame_length, 1)
    hop_length = max(hop_length, 1)
    frame_count = max(1, 1 + int(math.ceil(max(0, len(samples) - frame_length) / hop_length)))
    padded_length = (frame_count - 1) * hop_length + frame_length
    padded = np.pad(samples.astype(np.float64), (0, max(0, padded_length - len(samples))))
    starts = np.arange(frame_count) * hop_length
    frames = np.stack([padded[start:start + frame_length] for start in starts])
    return frames


def _pitch_from_frames(frames: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(frames)
    pitches = np.zeros(frame_count, dtype=np.float64)
    voiced = np.zeros(frame_count, dtype=bool)
    min_lag = max(1, int(sample_rate / 500.0))
    max_lag = min(frames.shape[1] - 1, int(sample_rate / 60.0))

    for index, frame in enumerate(frames):
        centered = frame - np.mean(frame)
        energy = float(np.dot(centered, centered))
        if energy <= 1e-8:
            continue

        corr = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
        reference = float(corr[0])
        if reference <= 1e-8 or max_lag <= min_lag:
            continue

        upper = corr[min_lag:max_lag + 1]
        lag = min_lag + int(np.argmax(upper))
        peak = float(corr[lag]) / reference
        if peak < 0.30:
            continue

        refined_lag = float(lag)
        if 1 <= lag < len(corr) - 1:
            y_left, y_center, y_right = corr[lag - 1], corr[lag], corr[lag + 1]
            denominator = y_left - 2.0 * y_center + y_right
            if abs(denominator) > 1e-12:
                refined_lag += 0.5 * (y_left - y_right) / denominator

        if refined_lag > 0:
            pitches[index] = sample_rate / refined_lag
            voiced[index] = True

    return pitches, voiced


def _frame_features(
    samples: np.ndarray,
    sample_rate: int,
    config: AcousticConfig,
) -> dict[str, np.ndarray]:
    frame_length = max(1, int(round(sample_rate * config.frame_length_ms / 1000.0)))
    hop_length = max(1, int(round(sample_rate * config.hop_length_ms / 1000.0)))
    if samples.size:
        emphasized = np.empty_like(samples, dtype=np.float32)
        emphasized[0] = samples[0]
        if len(samples) > 1:
            emphasized[1:] = samples[1:] - 0.97 * samples[:-1]
    else:
        emphasized = samples

    frames = _frame_signal(samples, frame_length, hop_length)
    emphasized_frames = _frame_signal(emphasized, frame_length, hop_length)
    window = np.hanning(frame_length)
    windowed = emphasized_frames * window

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    peak = np.max(np.abs(frames), axis=1)
    zero_crossing = np.mean(
        np.not_equal(np.signbit(frames[:, 1:]), np.signbit(frames[:, :-1])), axis=1
    ) if frame_length > 1 else np.zeros(len(frames))

    spectrum = np.abs(np.fft.rfft(windowed, n=config.n_fft))
    power = np.square(spectrum)
    frequencies = np.fft.rfftfreq(config.n_fft, d=1.0 / sample_rate)
    power_sum = np.sum(power, axis=1)

    centroid = np.sum(power * frequencies[None, :], axis=1) / np.maximum(power_sum, 1e-12)
    bandwidth = np.sqrt(
        np.sum(power * (frequencies[None, :] - centroid[:, None]) ** 2, axis=1)
        / np.maximum(power_sum, 1e-12)
    )
    cumulative = np.cumsum(power, axis=1)
    threshold = cumulative[:, -1:] * 0.85
    rolloff_indices = np.argmax(cumulative >= threshold, axis=1)
    rolloff = frequencies[rolloff_indices]

    log_power = np.log(np.maximum(power, 1e-12))
    flatness = np.exp(np.mean(log_power, axis=1)) / np.maximum(np.mean(power, axis=1), 1e-12)

    contrast_edges = np.array([200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, min(sample_rate / 2.0, 7600.0)])
    contrast_values: list[float] = []
    for low, high in zip(contrast_edges[:-1], contrast_edges[1:]):
        band = power[:, (frequencies >= low) & (frequencies < high)]
        if band.shape[1] < 2:
            contrast_values.append(0.0)
            continue
        high_energy = np.percentile(band, 90, axis=1)
        low_energy = np.percentile(band, 10, axis=1)
        contrast = 10.0 * np.log10(
            np.maximum(high_energy, 1e-12) / np.maximum(low_energy, 1e-12)
        )
        contrast_values.append(float(np.mean(contrast)))

    filters = _mel_filterbank(
        sample_rate,
        config.n_fft,
        config.n_mels,
        config.fmin_hz,
        config.fmax_hz,
    )
    mel_energy = power @ filters.T
    log_mel = np.log(np.maximum(mel_energy, 1e-10))
    mfcc = _dct_type_ii(log_mel, config.n_mfcc)

    pitches, voiced = _pitch_from_frames(frames, sample_rate)

    chroma = np.zeros(12, dtype=np.float64)
    for bin_index, frequency in enumerate(frequencies[1:], start=1):
        if frequency < 40.0:
            continue
        pitch_class = int(round(12.0 * math.log2(frequency / 440.0) + 69.0)) % 12
        chroma[pitch_class] += float(power[:, bin_index].mean())
    chroma /= max(float(chroma.sum()), 1e-12)

    return {
        "rms": rms,
        "peak": peak,
        "zcr": zero_crossing,
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness": flatness,
        "contrast": np.asarray(contrast_values, dtype=np.float64),
        "pitch": pitches,
        "voiced": voiced.astype(np.float64),
        "mfcc": mfcc,
        "chroma": chroma,
        "frame_length": np.array([frame_length], dtype=np.float64),
        "hop_length": np.array([hop_length], dtype=np.float64),
    }


def _estimate_voice_activity(rms: np.ndarray, config: AcousticConfig) -> np.ndarray:
    if rms.size == 0:
        return np.zeros(0, dtype=bool)

    noise_floor = float(np.percentile(rms, 20))
    signal_peak = float(np.percentile(rms, 95))

    if signal_peak <= config.min_voiced_rms:
        return np.zeros_like(rms, dtype=bool)

    dynamic_range = max(0.0, signal_peak - noise_floor)
    if dynamic_range < signal_peak * 0.05:
        threshold = max(config.min_voiced_rms, signal_peak * 0.25)
    else:
        threshold = max(
            config.min_voiced_rms,
            noise_floor + dynamic_range * config.silence_multiplier,
        )

    voiced = rms >= threshold

    if voiced.size >= 3:
        padded = np.pad(voiced.astype(np.int8), (1, 1), mode="edge")
        neighborhood = padded[:-2] + padded[1:-1] + padded[2:]
        voiced = neighborhood >= 2
    return voiced


def _count_runs(mask: np.ndarray, value: bool) -> int:
    if mask.size == 0:
        return 0
    changed = np.concatenate(([True], mask[1:] != mask[:-1]))
    starts = np.flatnonzero(changed)
    run_values = mask[starts]
    return int(np.sum(run_values == value))


def _estimate_syllable_rate(rms: np.ndarray, voiced_mask: np.ndarray, frame_seconds: float) -> float:
    if rms.size == 0 or not np.any(voiced_mask):
        return 0.0

    envelope = rms.copy()
    smooth_window = 5
    if len(envelope) >= smooth_window:
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        envelope = np.convolve(envelope, kernel, mode="same")

    voiced_values = envelope[voiced_mask]
    if voiced_values.size == 0:
        return 0.0

    threshold = max(float(np.percentile(voiced_values, 45)), 1e-6)
    peak_indices: list[int] = []
    min_gap_frames = max(1, int(round(0.18 / max(frame_seconds, 1e-6))))

    for index in range(1, len(envelope) - 1):
        if not voiced_mask[index]:
            continue
        if envelope[index] < threshold:
            continue
        if envelope[index] >= envelope[index - 1] and envelope[index] > envelope[index + 1]:
            if not peak_indices or index - peak_indices[-1] >= min_gap_frames:
                peak_indices.append(index)

    speech_seconds = max(float(voiced_mask.sum()) * frame_seconds, frame_seconds)
    return len(peak_indices) / speech_seconds


def _feature_vector(features: AcousticFeatures) -> np.ndarray:
    blocks: list[float] = []
    for name in (
        "pitch_hz",
        "energy",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_rolloff_hz",
        "spectral_flatness",
        "zero_crossing_rate",
    ):
        stats = getattr(features, name)
        blocks.extend([stats["mean"], stats["std"], stats["min"], stats["max"]])

    blocks.extend(features.spectral_contrast_db)
    blocks.extend(features.chroma)
    blocks.extend(features.mfcc_mean)
    blocks.extend(features.mfcc_std)
    blocks.extend(
        [
            features.duration_seconds,
            features.silence_ratio,
            features.speech_duration_seconds,
            features.estimated_syllable_rate_sps,
            features.estimated_pause_count,
        ]
    )
    return np.asarray(blocks, dtype=np.float32)


def extract_acoustic_features(
    samples: np.ndarray,
    sample_rate: int,
    config: AcousticConfig | None = None,
) -> AcousticFeatures:
    config = config or AcousticConfig()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    clean = np.asarray(samples, dtype=np.float32).reshape(-1)
    clean = np.nan_to_num(clean, nan=0.0, posinf=0.0, neginf=0.0)
    clean = np.clip(clean, -1.0, 1.0)

    features = _frame_features(clean, sample_rate, config)
    rms = features["rms"]
    voiced_mask = _estimate_voice_activity(rms, config)
    pitch = features["pitch"]
    voiced_pitch = pitch[voiced_mask & (pitch > 0)]

    frame_duration = config.hop_length_ms / 1000.0
    speech_duration = float(voiced_mask.sum() * frame_duration)
    duration = len(clean) / sample_rate
    pause_frames = int((~voiced_mask).sum())
    pause_ratio = pause_frames / max(len(voiced_mask), 1)
    pause_count = _count_runs(~voiced_mask, True)
    syllable_rate = _estimate_syllable_rate(rms, voiced_mask, frame_duration)

    voiced_mfcc = features["mfcc"][voiced_mask] if np.any(voiced_mask) else features["mfcc"]
    result = AcousticFeatures(
        version="acoustic-v1",
        sample_rate=sample_rate,
        duration_seconds=duration,
        frame_count=len(rms),
        frame_length_seconds=config.frame_length_ms / 1000.0,
        hop_length_seconds=config.hop_length_ms / 1000.0,
        voiced_frame_count=int(voiced_mask.sum()),
        silence_ratio=pause_ratio,
        speech_duration_seconds=speech_duration,
        estimated_syllable_rate_sps=syllable_rate,
        estimated_pause_count=pause_count,
        estimated_pause_ratio=pause_ratio,
        pitch_hz=_stats(voiced_pitch, positive_only=True),
        energy=_stats(rms),
        spectral_centroid_hz=_stats(features["centroid"]),
        spectral_bandwidth_hz=_stats(features["bandwidth"]),
        spectral_rolloff_hz=_stats(features["rolloff"]),
        spectral_flatness=_stats(features["flatness"]),
        spectral_contrast_db=features["contrast"].tolist(),
        zero_crossing_rate=_stats(features["zcr"]),
        chroma=features["chroma"].tolist(),
        mfcc_mean=np.mean(voiced_mfcc, axis=0).tolist(),
        mfcc_std=np.std(voiced_mfcc, axis=0).tolist(),
    )
    return result
