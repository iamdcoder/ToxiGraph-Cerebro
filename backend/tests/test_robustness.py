from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from app.robustness.augmentations import AudioAugmentationError, apply_augmentation, available_conditions


@pytest.fixture
def sample_wav(tmp_path: Path) -> Path:
    sample_rate = 16000
    duration = 2.0
    time = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    samples = (0.25 * np.sin(2 * np.pi * 220.0 * time)).astype(np.float32)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2").tobytes()
    path = tmp_path / "03-01-03-01-01-01-01.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return path


def _write_wav(path: Path, frequency: float, amplitude: float) -> None:
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=np.float64) / sample_rate
    samples = (amplitude * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    pcm = np.clip(samples * 32767.0, -32768, 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def test_available_conditions_include_clean_and_stress_families() -> None:
    conditions = available_conditions()
    assert conditions["clean"]["family"] == "baseline"
    assert "noise_10db" in conditions
    assert "reverb_strong" in conditions
    assert "lowpass_2500hz" in conditions
    assert len(conditions) >= 10


def test_gain_is_deterministic_and_clipped(sample_wav: Path) -> None:
    from app.audio.processor import AudioProcessor

    processed = AudioProcessor().process(sample_wav.read_bytes())
    first = apply_augmentation(processed.samples, processed.sample_rate, "gain_up_6db")
    second = apply_augmentation(processed.samples, processed.sample_rate, "gain_up_6db")
    assert np.allclose(first, second)
    assert np.max(np.abs(first)) <= 1.0


def test_noise_changes_signal_but_preserves_length(sample_wav: Path) -> None:
    from app.audio.processor import AudioProcessor

    processed = AudioProcessor().process(sample_wav.read_bytes())
    noisy = apply_augmentation(processed.samples, processed.sample_rate, "noise_10db", seed=7)
    assert noisy.size == processed.samples.size
    assert noisy.ndim == 1
    assert not np.allclose(noisy, processed.samples.reshape(-1))


def test_time_stretch_changes_length(sample_wav: Path) -> None:
    from app.audio.processor import AudioProcessor

    processed = AudioProcessor().process(sample_wav.read_bytes())
    slower = apply_augmentation(processed.samples, processed.sample_rate, "speed_90pct")
    faster = apply_augmentation(processed.samples, processed.sample_rate, "speed_110pct")
    assert slower.size > processed.samples.size
    assert faster.size < processed.samples.size


def test_invalid_condition_and_bad_parameters_are_rejected(sample_wav: Path) -> None:
    from app.audio.processor import AudioProcessor

    processed = AudioProcessor().process(sample_wav.read_bytes())
    with pytest.raises(AudioAugmentationError):
        apply_augmentation(processed.samples, processed.sample_rate, "not_real")
    with pytest.raises(AudioAugmentationError):
        from app.robustness.augmentations import lowpass
        lowpass(processed.samples, processed.sample_rate, processed.sample_rate / 2)


def test_robustness_benchmark_produces_clean_and_condition_metrics(tmp_path: Path) -> None:
    from app.robustness.benchmark import run_robustness_benchmark
    paths = []
    records = []
    for index, (freq, label, speaker) in enumerate(
        [(180, "neutral", "s1"), (260, "happy", "s1"), (180, "neutral", "s2"), (260, "happy", "s2")]
    ):
        path = tmp_path / f"sample-{index}.wav"
        _write_wav(path, freq, 0.30)
        paths.append(path)
        records.append({"audio_path": str(path), "label": label, "speaker_id": speaker})

    def predictor(samples, sample_rate):
        del sample_rate
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        happy = min(0.95, max(0.05, rms / 0.30))
        return {"neutral": 1.0 - happy, "happy": happy}

    result = run_robustness_benchmark(
        records,
        predictor,
        conditions=["clean", "noise_10db", "gain_down_12db"],
    )
    payload = result.to_dict()
    assert payload["records"] == 4
    assert payload["clean_metrics"]["macro_f1"] >= 0.0
    assert len(payload["conditions"]) == 3
    assert payload["integrity"]["speaker_leakage_checked"] is True


def test_robustness_requires_speaker_id_and_audio_path() -> None:
    from app.robustness.benchmark import RobustnessBenchmarkError, run_robustness_benchmark
    def predictor(samples, sample_rate):
        del samples, sample_rate
        return {"neutral": 1.0}

    with pytest.raises(RobustnessBenchmarkError):
        run_robustness_benchmark([{"label": "neutral", "audio_path": "x.wav"}], predictor)
    with pytest.raises(RobustnessBenchmarkError):
        run_robustness_benchmark([{"label": "neutral", "speaker_id": "s1"}], predictor)

