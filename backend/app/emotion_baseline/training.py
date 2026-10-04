from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.audio.processor import AudioProcessor
from app.features.acoustic import FEATURE_VECTOR_NAMES, extract_acoustic_features
from app.datasets.ravdess import RAVDESSSample, build_ravdess_manifest


SEED = 42


@dataclass(frozen=True)
class SplitAssignment:
    train: tuple[RAVDESSSample, ...]
    validation: tuple[RAVDESSSample, ...]
    test: tuple[RAVDESSSample, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "train_samples": len(self.train),
            "validation_samples": len(self.validation),
            "test_samples": len(self.test),
            "train_speakers": sorted({sample.speaker_id for sample in self.train}),
            "validation_speakers": sorted({sample.speaker_id for sample in self.validation}),
            "test_speakers": sorted({sample.speaker_id for sample in self.test}),
        }


def speaker_independent_split(samples: list[RAVDESSSample], seed: int = SEED) -> SplitAssignment:
    if len({sample.speaker_id for sample in samples}) < 6:
        raise ValueError("Speaker-independent splitting requires at least 6 speakers.")

    groups = np.asarray([sample.speaker_id for sample in samples])
    indices = np.arange(len(samples))

    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_val_idx, test_idx = next(first.split(indices, groups=groups))

    train_val_samples = [samples[index] for index in train_val_idx]
    train_val_groups = np.asarray([sample.speaker_id for sample in train_val_samples])
    train_val_indices = np.arange(len(train_val_samples))

    second = GroupShuffleSplit(n_splits=1, test_size=0.1764705882353, random_state=seed)
    train_idx, val_idx = next(second.split(train_val_indices, groups=train_val_groups))

    train = tuple(train_val_samples[index] for index in train_idx)
    validation = tuple(train_val_samples[index] for index in val_idx)
    test = tuple(samples[index] for index in test_idx)

    speaker_sets = [
        {sample.speaker_id for sample in train},
        {sample.speaker_id for sample in validation},
        {sample.speaker_id for sample in test},
    ]
    if speaker_sets[0] & speaker_sets[1] or speaker_sets[0] & speaker_sets[2] or speaker_sets[1] & speaker_sets[2]:
        raise AssertionError("Speaker leakage detected between data splits.")

    return SplitAssignment(train=train, validation=validation, test=test)


def _extract(samples: tuple[RAVDESSSample, ...]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    processor = AudioProcessor()
    rows: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    for sample in samples:
        audio = processor.process(sample.path.read_bytes())
        features = extract_acoustic_features(audio.samples, audio.sample_rate)
        vector = features.to_vector().astype(np.float64)
        if vector.shape != (len(FEATURE_VECTOR_NAMES),) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Invalid feature vector for {sample.path}")
        rows.append(vector)
        labels.append(sample.emotion)
        paths.append(str(sample.path))

    return np.vstack(rows), np.asarray(labels), paths


def _make_candidates() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
        ]),
        "linear_svm": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", SVC(C=2.0, class_weight="balanced", probability=True, random_state=SEED)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def _metric_report(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {
            label: float(score)
            for label, score in zip(
                labels,
                f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0),
                strict=True,
            )
        },
        "confusion_matrix": matrix.astype(int).tolist(),
        "labels": labels,
    }


def _write_manifest(path: Path, split_name: str, samples: tuple[RAVDESSSample, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for sample in samples:
            writer.writerow([split_name, str(sample.path), sample.speaker_id, sample.emotion])


def train_baseline(
    data_dir: str | Path,
    output_model: str | Path,
    output_dir: str | Path | None = None,
    seed: int = SEED,
) -> dict[str, Any]:
    samples = build_ravdess_manifest(data_dir)
    split = speaker_independent_split(samples, seed=seed)

    if len(split.train) == 0 or len(split.validation) == 0 or len(split.test) == 0:
        raise ValueError("Train, validation and test splits must all be non-empty.")

    X_train, y_train, train_paths = _extract(split.train)
    X_val, y_val, val_paths = _extract(split.validation)
    X_test, y_test, test_paths = _extract(split.test)

    labels = sorted({sample.emotion for sample in samples})
    label_to_index = {label: index for index, label in enumerate(labels)}
    y_train_encoded = np.asarray([label_to_index[label] for label in y_train], dtype=np.int64)
    y_val_encoded = np.asarray([label_to_index[label] for label in y_val], dtype=np.int64)
    y_test_encoded = np.asarray([label_to_index[label] for label in y_test], dtype=np.int64)

    candidate_results: dict[str, dict[str, Any]] = {}
    candidates = _make_candidates()
    for name, model in candidates.items():
        model.fit(X_train, y_train_encoded)
        validation_pred = model.predict(X_val).astype(int)
        candidate_results[name] = {
            "validation": _metric_report(y_val_encoded, validation_pred, list(range(len(labels)))),
        }

    selected_name = max(
        candidate_results,
        key=lambda name: (
            candidate_results[name]["validation"]["macro_f1"],
            candidate_results[name]["validation"]["accuracy"],
            name,
        ),
    )

    selected_model = candidates[selected_name]
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train_encoded, y_val_encoded])
    selected_model.fit(X_train_val, y_train_val)

    test_pred = selected_model.predict(X_test).astype(int)
    test_report = _metric_report(y_test_encoded, test_pred, list(range(len(labels))))

    output_model_path = Path(output_model).expanduser().resolve()
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    training_metadata = {
        "dataset": "RAVDESS Speech",
        "dataset_source": "https://zenodo.org/records/1188976",
        "dataset_variant": "RAVDESS Speech 16K",
        "dataset_variant_source": "https://zenodo.org/records/11063852",
        "seed": seed,
        "feature_version": "acoustic-v1",
        "feature_count": len(FEATURE_VECTOR_NAMES),
        "labels": labels,
        "split": split.summary(),
        "class_distribution": {
            "all": dict(Counter(sample.emotion for sample in samples)),
            "train": dict(Counter(sample.emotion for sample in split.train)),
            "validation": dict(Counter(sample.emotion for sample in split.validation)),
            "test": dict(Counter(sample.emotion for sample in split.test)),
        },
        "selection_rule": "highest validation macro-F1, then validation accuracy, deterministic name tie-break",
        "candidate_validation": candidate_results,
        "test": test_report,
    }

    bundle = {
        "model_name": selected_name,
        "model": selected_model,
        "labels": labels,
        "feature_names": FEATURE_VECTOR_NAMES,
        "feature_version": "acoustic-v1",
        "training_metadata": training_metadata,
    }
    joblib.dump(bundle, output_model_path)

    result = {
        "success": True,
        "model_path": str(output_model_path),
        "selected_model": selected_name,
        "feature_count": len(FEATURE_VECTOR_NAMES),
        "split": split.summary(),
        "candidate_validation": candidate_results,
        "test": test_report,
    }

    if output_dir is not None:
        out = Path(output_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "training_metadata.json").write_text(json.dumps(training_metadata, indent=2), encoding="utf-8")
        manifest_path = out / "manifest.csv"
        manifest_path.write_text("split,path,speaker_id,emotion\n", encoding="utf-8")
        _write_manifest(manifest_path, "train", split.train)
        _write_manifest(manifest_path, "validation", split.validation)
        _write_manifest(manifest_path, "test", split.test)

    return result
