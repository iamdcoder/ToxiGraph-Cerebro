from __future__ import annotations

import io
import wave
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.metrics import f1_score

from app.datasets.ravdess import RAVDESS_EMOTIONS
from app.emotion_baseline.model import ClassicalEmotionModel
from app.features.acoustic import FEATURE_VECTOR_NAMES
from app.optimization.classical import (
    OptimizationError,
    FeatureDataset,
    build_optimization_candidates,
    load_feature_cache,
    optimize_classical_baseline,
    save_feature_cache,
)


def _write_tone(path: Path, freq: float) -> None:
    sr = 16000
    duration = 0.55
    t = np.arange(int(sr * duration), dtype=np.float32) / sr
    samples = 0.2 * np.sin(2 * np.pi * freq * t)
    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sr)
            wav.writeframes((samples * 32767).astype(np.int16).tobytes())
        path.write_bytes(buffer.getvalue())


def _make_dataset() -> FeatureDataset:
    rng = np.random.default_rng(25)
    labels = np.asarray([RAVDESS_EMOTIONS[index] for index in range(1, 8) for _ in range(12)])[:72]
    speakers = np.asarray([f"S{speaker:02d}" for speaker in range(1, 25) for _ in range(3)])
    X = rng.normal(0, 1, size=(len(labels), len(FEATURE_VECTOR_NAMES)))
    return FeatureDataset(
        X=X,
        labels=labels,
        speakers=speakers,
        paths=tuple(f"sample-{index}.wav" for index in range(len(labels))),
    )


def test_feature_cache_round_trip(tmp_path: Path):
    dataset = _make_dataset()
    path = save_feature_cache(dataset, tmp_path / "features.npz")
    loaded = load_feature_cache(path)
    assert loaded.X.shape == dataset.X.shape
    assert loaded.labels.tolist() == dataset.labels.tolist()
    assert loaded.speakers.tolist() == dataset.speakers.tolist()


def test_invalid_cache_version_is_rejected(tmp_path: Path):
    payload = {
        "feature_version": "wrong",
        "feature_names": list(FEATURE_VECTOR_NAMES),
        "labels": ["angry"],
        "speakers": ["S01"],
        "paths": ["a.wav"],
    }
    np.savez_compressed(tmp_path / "bad.npz", X=np.zeros((1, len(FEATURE_VECTOR_NAMES))), metadata=__import__("json").dumps(payload))
    with pytest.raises(OptimizationError):
        load_feature_cache(tmp_path / "bad.npz")


def test_candidate_grids_have_expected_models():
    candidates = build_optimization_candidates()
    assert set(candidates) == {"logistic_regression", "linear_svm", "random_forest"}
    for estimator, grid in candidates.values():
        assert grid
        assert hasattr(estimator, "fit")


def test_optimizer_rejects_invalid_cv_configuration(tmp_path: Path, monkeypatch):
    with pytest.raises(OptimizationError):
        optimize_classical_baseline(tmp_path, tmp_path / "model.joblib", tmp_path / "out", cv_folds=1)


def test_optimizer_produces_loadable_artifact_from_cached_fixture(tmp_path: Path, monkeypatch):
    from app.optimization import classical
    from app.emotion_baseline.training import SplitAssignment

    dataset = _make_dataset()
    cache_path = save_feature_cache(dataset, tmp_path / "features.npz")
    samples = []

    class Sample:
        def __init__(self, index: int):
            self.speaker_id = dataset.speakers[index]
            self.path = Path(dataset.paths[index])
            self.emotion = str(dataset.labels[index])

    samples = [Sample(index) for index in range(len(dataset.X))]
    train = tuple(samples[:48])
    val = tuple(samples[48:60])
    test = tuple(samples[60:72])
    monkeypatch.setattr(classical, "build_ravdess_manifest", lambda _: samples)
    monkeypatch.setattr(classical, "speaker_independent_split", lambda _samples, seed=42: SplitAssignment(train, val, test))

    result = classical.optimize_classical_baseline(
        tmp_path,
        tmp_path / "optimized.joblib",
        tmp_path / "out",
        cv_folds=3,
        feature_cache=cache_path,
    )
    assert result["success"] is True
    assert result["selected_model"] in {"logistic_regression", "linear_svm", "random_forest"}
    assert 0.0 <= result["test"]["macro_f1"] <= 1.0
    model = ClassicalEmotionModel.load(tmp_path / "optimized.joblib")
    assert model.metadata()["feature_version"] == "acoustic-v1"
