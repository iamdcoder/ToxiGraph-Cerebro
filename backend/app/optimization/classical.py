from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.audio.processor import AudioProcessor
from app.datasets.ravdess import RAVDESSSample, build_ravdess_manifest
from app.emotion_baseline.model import ClassicalEmotionModel
from app.emotion_baseline.training import SEED, speaker_independent_split
from app.features.acoustic import FEATURE_VECTOR_NAMES, extract_acoustic_features
from app.fusion.constants import CANONICAL_LABELS, normalize_emotion_label


class OptimizationError(ValueError):
    """Raised when model optimization inputs are invalid."""


@dataclass(frozen=True)
class FeatureDataset:
    X: np.ndarray
    labels: np.ndarray
    speakers: np.ndarray
    paths: tuple[str, ...]

    def validate(self) -> None:
        if self.X.ndim != 2 or self.X.shape[1] != len(FEATURE_VECTOR_NAMES):
            raise OptimizationError("Feature matrix has an invalid shape.")
        if len(self.labels) != len(self.X) or len(self.speakers) != len(self.X):
            raise OptimizationError("Feature metadata lengths do not match feature rows.")
        if not np.all(np.isfinite(self.X)):
            raise OptimizationError("Feature matrix contains non-finite values.")
        if len(self.X) == 0:
            raise OptimizationError("Feature dataset is empty.")


def extract_feature_dataset(samples: tuple[RAVDESSSample, ...]) -> FeatureDataset:
    processor = AudioProcessor()
    rows: list[np.ndarray] = []
    labels: list[str] = []
    speakers: list[str] = []
    paths: list[str] = []
    for sample in samples:
        processed = processor.process(sample.path.read_bytes())
        if processed.is_silent:
            raise OptimizationError(f"Cannot optimize with silent sample: {sample.path}")
        features = extract_acoustic_features(processed.samples, processed.sample_rate)
        vector = np.asarray(features.to_vector(), dtype=np.float64)
        if vector.shape != (len(FEATURE_VECTOR_NAMES),) or not np.all(np.isfinite(vector)):
            raise OptimizationError(f"Invalid acoustic feature vector: {sample.path}")
        rows.append(vector)
        labels.append(sample.emotion)
        speakers.append(str(sample.speaker_id))
        paths.append(str(sample.path))

    dataset = FeatureDataset(
        X=np.vstack(rows),
        labels=np.asarray(labels),
        speakers=np.asarray(speakers),
        paths=tuple(paths),
    )
    dataset.validate()
    return dataset


def save_feature_cache(dataset: FeatureDataset, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "feature_version": "acoustic-v1",
        "feature_names": list(FEATURE_VECTOR_NAMES),
        "records": len(dataset.X),
        "paths": list(dataset.paths),
        "labels": dataset.labels.tolist(),
        "speakers": dataset.speakers.tolist(),
    }
    np.savez_compressed(destination, X=dataset.X, metadata=json.dumps(metadata))
    return destination


def load_feature_cache(path: str | Path) -> FeatureDataset:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Feature cache does not exist: {source}")
    try:
        payload = np.load(source, allow_pickle=False)
        X = np.asarray(payload["X"], dtype=np.float64)
        metadata = json.loads(str(payload["metadata"]))
    except Exception as exc:
        raise OptimizationError(f"Could not load feature cache: {source}") from exc
    if metadata.get("feature_version") != "acoustic-v1":
        raise OptimizationError("Feature cache uses an incompatible feature version.")
    if metadata.get("feature_names") != list(FEATURE_VECTOR_NAMES):
        raise OptimizationError("Feature cache feature names do not match the active extractor.")
    dataset = FeatureDataset(
        X=X,
        labels=np.asarray(metadata.get("labels", [])),
        speakers=np.asarray(metadata.get("speakers", [])),
        paths=tuple(str(item) for item in metadata.get("paths", [])),
    )
    dataset.validate()
    return dataset


def build_optimization_candidates(seed: int = SEED) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "logistic_regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=2500, class_weight="balanced", random_state=seed)),
            ]),
            {
                "classifier__C": [0.1, 0.5, 1.0, 2.0, 5.0],
                "classifier__solver": ["lbfgs"],
            },
        ),
        "linear_svm": (
            Pipeline([
                ("scale", StandardScaler()),
                ("classifier", SVC(class_weight="balanced", probability=True, random_state=seed)),
            ]),
            {
                "classifier__C": [0.25, 0.5, 1.0, 2.0, 4.0],
                "classifier__kernel": ["linear"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=1,
            ),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 12, 20],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", 0.5],
            },
        ),
    }


def _encode_labels(labels: np.ndarray, classes: tuple[str, ...] = tuple(CANONICAL_LABELS)) -> np.ndarray:
    index = {label: i for i, label in enumerate(classes)}
    normalized = [normalize_emotion_label(str(label)) for label in labels]
    unknown = sorted({label for label in normalized if label not in index})
    if unknown:
        raise OptimizationError(f"Unknown emotion labels: {unknown}")
    return np.asarray([index[label] for label in normalized], dtype=np.int64)


def _quality_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _top_cv_rows(search: GridSearchCV, limit: int = 8) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    means = np.asarray(search.cv_results_["mean_test_score"], dtype=np.float64)
    stds = np.asarray(search.cv_results_["std_test_score"], dtype=np.float64)
    params = search.cv_results_["params"]
    ranks = np.asarray(search.cv_results_["rank_test_score"], dtype=np.int64)
    order = sorted(range(len(params)), key=lambda i: (int(ranks[i]), -float(means[i]), json.dumps(params[i], sort_keys=True)))
    for index in order[:limit]:
        table.append({
            "rank": int(ranks[index]),
            "mean_macro_f1": float(means[index]),
            "std_macro_f1": float(stds[index]),
            "params": params[index],
        })
    return table


def optimize_classical_baseline(
    data_dir: str | Path,
    output_model: str | Path,
    output_dir: str | Path,
    *,
    seed: int = SEED,
    cv_folds: int = 4,
    feature_cache: str | Path | None = None,
) -> dict[str, Any]:
    if cv_folds < 2:
        raise OptimizationError("cv_folds must be at least 2.")

    samples = build_ravdess_manifest(data_dir)
    split = speaker_independent_split(samples, seed=seed)
    train_speakers = {sample.speaker_id for sample in split.train}
    if len(train_speakers) < cv_folds:
        raise OptimizationError(
            f"Training split contains {len(train_speakers)} speakers; cannot use {cv_folds}-fold grouped CV."
        )

    cache_path = Path(feature_cache).expanduser().resolve() if feature_cache else None
    if cache_path and cache_path.exists():
        all_features = load_feature_cache(cache_path)
    else:
        all_features = extract_feature_dataset(tuple(samples))
        if cache_path:
            save_feature_cache(all_features, cache_path)

    by_path = {path: index for index, path in enumerate(all_features.paths)}
    train_idx = [by_path[str(sample.path)] for sample in split.train]
    val_idx = [by_path[str(sample.path)] for sample in split.validation]
    test_idx = [by_path[str(sample.path)] for sample in split.test]

    X_train = all_features.X[train_idx]
    X_val = all_features.X[val_idx]
    X_test = all_features.X[test_idx]
    y_train = _encode_labels(all_features.labels[train_idx])
    y_val = _encode_labels(all_features.labels[val_idx])
    y_test = _encode_labels(all_features.labels[test_idx])
    groups_train = all_features.speakers[train_idx]

    actual_cv = min(cv_folds, len(np.unique(groups_train)))
    cv = GroupKFold(n_splits=actual_cv)

    candidate_results: dict[str, Any] = {}
    best_by_validation: list[tuple[str, float, float]] = []

    for name, (estimator, param_grid) in build_optimization_candidates(seed).items():
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
            return_train_score=False,
        )
        search.fit(X_train, y_train, groups=groups_train)
        val_pred = search.best_estimator_.predict(X_val)
        validation = _quality_metrics(y_val, val_pred)
        candidate_results[name] = {
            "best_params": search.best_params_,
            "cv": {
                "folds": actual_cv,
                "best_macro_f1": float(search.best_score_),
                "top_configurations": _top_cv_rows(search),
            },
            "validation": validation,
        }
        best_by_validation.append((name, validation["macro_f1"], validation["accuracy"]))

    selected_name = max(best_by_validation, key=lambda item: (item[1], item[2], item[0]))[0]
    selected_params = candidate_results[selected_name]["best_params"]

    candidates = build_optimization_candidates(seed)
    selected_estimator = candidates[selected_name][0].set_params(**selected_params)
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    selected_estimator.fit(X_train_val, y_train_val)

    test_pred = selected_estimator.predict(X_test)
    test_metrics = _quality_metrics(y_test, test_pred)

    output_model_path = Path(output_model).expanduser().resolve()
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "optimizer_version": "classical-opt-v1",
        "dataset": "RAVDESS Speech",
        "dataset_source": "https://zenodo.org/records/1188976",
        "dataset_variant": "RAVDESS Speech 16K",
        "dataset_variant_source": "https://zenodo.org/records/11063852",
        "seed": seed,
        "feature_version": "acoustic-v1",
        "feature_count": len(FEATURE_VECTOR_NAMES),
        "cv_strategy": "GroupKFold by speaker on training split only",
        "cv_folds": actual_cv,
        "selection_rule": "highest validation macro-F1, then validation accuracy, deterministic name tie-break",
        "split": split.summary(),
        "selected_model": selected_name,
        "selected_params": selected_params,
        "candidate_results": candidate_results,
        "test": test_metrics,
        "feature_cache": str(cache_path) if cache_path else None,
    }

    joblib.dump({
        "model_name": selected_name,
        "model": selected_estimator,
        "labels": list(CANONICAL_LABELS),
        "feature_names": FEATURE_VECTOR_NAMES,
        "feature_version": "acoustic-v1",
        "training_metadata": metadata,
    }, output_model_path)

    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "success": True,
        "model_path": str(output_model_path),
        "selected_model": selected_name,
        "selected_params": selected_params,
        "cv": {"strategy": metadata["cv_strategy"], "folds": actual_cv},
        "candidate_results": candidate_results,
        "test": test_metrics,
        "split": split.summary(),
        "feature_count": len(FEATURE_VECTOR_NAMES),
    }
    (out / "optimization.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if cache_path:
        (out / "feature_cache.json").write_text(
            json.dumps({"path": str(cache_path), "records": len(all_features.X)}, indent=2),
            encoding="utf-8",
        )
    return result


def load_optimized_model(path: str | Path) -> ClassicalEmotionModel:
    return ClassicalEmotionModel.load(path)
