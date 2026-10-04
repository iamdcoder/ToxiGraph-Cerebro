from __future__ import annotations

import math
from typing import Callable

import numpy as np


class AudioAugmentationError(ValueError):
    """Raised when a robustness perturbation cannot be applied safely."""


Augmentation = Callable[[np.ndarray, int, int], np.ndarray]


def _validate(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    if array.size == 0:
        raise AudioAugmentationError("Audio augmentation requires non-empty samples.")
    if sample_rate <= 0:
        raise AudioAugmentationError("Sample rate must be positive.")
    if not np.isfinite(array).all():
        raise AudioAugmentationError("Audio augmentation received non-finite samples.")
    return np.clip(array, -1.0, 1.0)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gain_db(samples: np.ndarray, sample_rate: int, amount_db: float, seed: int = 0) -> np.ndarray:
    del seed
    array = _validate(samples, sample_rate)
    factor = 10.0 ** (float(amount_db) / 20.0)
    return np.clip(array * factor, -1.0, 1.0).astype(np.float32)


def additive_noise_db(samples: np.ndarray, sample_rate: int, snr_db: float, seed: int = 0) -> np.ndarray:
    array = _validate(samples, sample_rate)
    clean_rms = float(np.sqrt(np.mean(np.square(array), dtype=np.float64)))
    if clean_rms <= 1e-8:
        raise AudioAugmentationError("Cannot add SNR-controlled noise to silent audio.")
    noise = _rng(seed).normal(0.0, 1.0, size=array.shape).astype(np.float32)
    noise -= float(np.mean(noise))
    noise_rms = float(np.sqrt(np.mean(np.square(noise), dtype=np.float64)))
    target_noise_rms = clean_rms / (10.0 ** (float(snr_db) / 20.0))
    noise *= target_noise_rms / max(noise_rms, 1e-8)
    return np.clip(array + noise, -1.0, 1.0).astype(np.float32)


def hard_clip(samples: np.ndarray, sample_rate: int, threshold: float, seed: int = 0) -> np.ndarray:
    del seed
    array = _validate(samples, sample_rate)
    threshold = float(threshold)
    if not 0.0 < threshold <= 1.0:
        raise AudioAugmentationError("Clipping threshold must be in (0, 1].")
    return np.clip(array, -threshold, threshold).astype(np.float32)


def time_stretch(samples: np.ndarray, sample_rate: int, factor: float, seed: int = 0) -> np.ndarray:
    del seed
    array = _validate(samples, sample_rate)
    factor = float(factor)
    if factor <= 0.0 or not math.isfinite(factor):
        raise AudioAugmentationError("Time-stretch factor must be finite and positive.")
    target_length = max(1, int(round(array.size / factor)))
    source_positions = np.arange(array.size, dtype=np.float64)
    target_positions = np.linspace(0.0, max(0.0, array.size - 1), target_length, dtype=np.float64)
    stretched = np.interp(target_positions, source_positions, array.astype(np.float64))
    return np.clip(stretched, -1.0, 1.0).astype(np.float32)


def lowpass(samples: np.ndarray, sample_rate: int, cutoff_hz: float, seed: int = 0) -> np.ndarray:
    del seed
    array = _validate(samples, sample_rate)
    cutoff_hz = float(cutoff_hz)
    if cutoff_hz <= 0.0 or cutoff_hz >= sample_rate / 2.0:
        raise AudioAugmentationError("Low-pass cutoff must lie strictly between 0 and Nyquist.")
    dt = 1.0 / sample_rate
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    alpha = dt / (rc + dt)
    output = np.empty_like(array)
    output[0] = array[0]
    for index in range(1, array.size):
        output[index] = output[index - 1] + alpha * (array[index] - output[index - 1])
    return np.clip(output, -1.0, 1.0).astype(np.float32)


def reverberate(samples: np.ndarray, sample_rate: int, decay: float, delay_ms: float, seed: int = 0) -> np.ndarray:
    del seed
    array = _validate(samples, sample_rate)
    decay = float(decay)
    delay_ms = float(delay_ms)
    if not 0.0 <= decay <= 1.0:
        raise AudioAugmentationError("Reverb decay must be between 0 and 1.")
    delay_samples = max(1, int(round(sample_rate * delay_ms / 1000.0)))
    if delay_samples >= array.size:
        return array.copy()
    output = np.array(array, copy=True, dtype=np.float32)
    for step, multiplier in enumerate((0.55, 0.35, 0.20), start=1):
        delay = delay_samples * step
        if delay >= array.size:
            continue
        output[delay:] += (decay * multiplier) * array[:-delay]
    peak = float(np.max(np.abs(output)))
    if peak > 1.0:
        output /= peak
    return np.clip(output, -1.0, 1.0).astype(np.float32)


def quantize(samples: np.ndarray, sample_rate: int, bits: int, seed: int = 0) -> np.ndarray:
    del seed
    array = _validate(samples, sample_rate)
    bits = int(bits)
    if bits < 2 or bits > 24:
        raise AudioAugmentationError("Quantization bit depth must be between 2 and 24.")
    levels = float((2 ** (bits - 1)) - 1)
    quantized = np.round(array * levels) / levels
    return np.clip(quantized, -1.0, 1.0).astype(np.float32)


def available_conditions() -> dict[str, dict[str, float | int | str]]:
    return {
        "clean": {"family": "baseline", "description": "Unmodified reference audio."},
        "gain_down_12db": {"family": "level", "amount_db": -12.0, "description": "Low microphone level."},
        "gain_up_6db": {"family": "level", "amount_db": 6.0, "description": "High input gain."},
        "noise_20db": {"family": "noise", "snr_db": 20.0, "description": "Moderate additive broadband noise."},
        "noise_10db": {"family": "noise", "snr_db": 10.0, "description": "Strong additive broadband noise."},
        "noise_5db": {"family": "noise", "snr_db": 5.0, "description": "Very strong additive broadband noise."},
        "clip_0_5pct": {"family": "clipping", "threshold": 0.85, "description": "Moderate waveform clipping stress."},
        "clip_1pct": {"family": "clipping", "threshold": 0.70, "description": "Severe waveform clipping stress."},
        "speed_90pct": {"family": "speed", "factor": 0.90, "description": "Speech slowed by 10%."},
        "speed_110pct": {"family": "speed", "factor": 1.10, "description": "Speech sped up by 10%."},
        "lowpass_2500hz": {"family": "bandwidth", "cutoff_hz": 2500.0, "description": "Reduced high-frequency bandwidth."},
        "reverb_light": {"family": "reverb", "decay": 0.55, "delay_ms": 55.0, "description": "Mild room-like reverberation."},
        "reverb_strong": {"family": "reverb", "decay": 0.80, "delay_ms": 85.0, "description": "Strong reverberation."},
        "quantize_8bit": {"family": "quantization", "bits": 8, "description": "Low-bit-depth quantization."},
    }


def apply_augmentation(samples: np.ndarray, sample_rate: int, condition: str, *, seed: int = 42) -> np.ndarray:
    conditions = available_conditions()
    if condition not in conditions:
        raise AudioAugmentationError(f"Unknown robustness condition: {condition}")
    spec = conditions[condition]
    if condition == "clean":
        return _validate(samples, sample_rate).copy()
    family = spec["family"]
    if family == "level":
        return gain_db(samples, sample_rate, float(spec["amount_db"]), seed)
    if family == "noise":
        return additive_noise_db(samples, sample_rate, float(spec["snr_db"]), seed)
    if family == "clipping":
        return hard_clip(samples, sample_rate, float(spec["threshold"]), seed)
    if family == "speed":
        return time_stretch(samples, sample_rate, float(spec["factor"]), seed)
    if family == "bandwidth":
        return lowpass(samples, sample_rate, float(spec["cutoff_hz"]), seed)
    if family == "reverb":
        return reverberate(samples, sample_rate, float(spec["decay"]), float(spec["delay_ms"]), seed)
    if family == "quantization":
        return quantize(samples, sample_rate, int(spec["bits"]), seed)
    raise AudioAugmentationError(f"Unsupported robustness family: {family}")
