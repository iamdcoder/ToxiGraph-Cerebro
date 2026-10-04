from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import ClassifierMixin


@dataclass(frozen=True)
class BaselineBundle:
    model_name: str
    model: ClassifierMixin
    labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    feature_version: str
    training_metadata: dict[str, Any]


class BaselineModelError(RuntimeError):
    """Raised when a trained baseline model cannot be loaded or used."""


class ClassicalEmotionModel:
    def __init__(self, bundle: BaselineBundle) -> None:
        self.bundle = bundle

    @classmethod
    def load(cls, path: str | Path) -> "ClassicalEmotionModel":
        model_path = Path(path).expanduser().resolve()
        if not model_path.exists():
            raise BaselineModelError(
                f"Classical emotion model is not trained yet: {model_path}"
            )
        try:
            payload = joblib.load(model_path)
        except Exception as exc:
            raise BaselineModelError(
                f"Could not load classical emotion model: {model_path}"
            ) from exc

        required = {
            "model_name",
            "model",
            "labels",
            "feature_names",
            "feature_version",
            "training_metadata",
        }
        missing = required.difference(payload)
        if missing:
            raise BaselineModelError(
                f"Classical emotion model is missing metadata: {sorted(missing)}"
            )

        return cls(
            BaselineBundle(
                model_name=str(payload["model_name"]),
                model=payload["model"],
                labels=tuple(str(label) for label in payload["labels"]),
                feature_names=tuple(str(name) for name in payload["feature_names"]),
                feature_version=str(payload["feature_version"]),
                training_metadata=dict(payload["training_metadata"]),
            )
        )

    def predict_proba(self, vector: np.ndarray) -> dict[str, float]:
        array = np.asarray(vector, dtype=np.float64)
        expected = len(self.bundle.feature_names)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != expected:
            raise BaselineModelError(
                f"Expected {expected} acoustic features, received shape {array.shape}."
            )
        if not np.all(np.isfinite(array)):
            raise BaselineModelError("Acoustic feature vector contains non-finite values.")

        if not hasattr(self.bundle.model, "predict_proba"):
            raise BaselineModelError("The trained classical model does not expose probabilities.")

        raw = np.asarray(self.bundle.model.predict_proba(array), dtype=np.float64)[0]
        model_classes = getattr(self.bundle.model, "classes_", np.arange(len(raw)))
        labels = self.bundle.labels
        probabilities = {label: 0.0 for label in labels}
        for index, class_value in enumerate(model_classes):
            class_index = int(class_value)
            if 0 <= class_index < len(labels):
                probabilities[labels[class_index]] = float(raw[index])

        total = sum(probabilities.values())
        if total <= 0:
            raise BaselineModelError("The trained model returned an invalid probability distribution.")
        return {label: value / total for label, value in probabilities.items()}

    def predict(self, vector: np.ndarray) -> dict[str, Any]:
        probabilities = self.predict_proba(vector)
        predicted = max(probabilities, key=probabilities.get)
        confidence = float(probabilities[predicted])
        return {
            "emotion": predicted,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_name": self.bundle.model_name,
            "feature_version": self.bundle.feature_version,
            "training_metadata": self.bundle.training_metadata,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.bundle.model_name,
            "feature_version": self.bundle.feature_version,
            "feature_count": len(self.bundle.feature_names),
            "labels": list(self.bundle.labels),
            "training_metadata": self.bundle.training_metadata,
        }
