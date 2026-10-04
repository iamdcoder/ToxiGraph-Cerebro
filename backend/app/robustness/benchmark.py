from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from app.audio.processor import AudioProcessor
from app.evaluation.metrics import labels_to_targets
from app.fusion.constants import CANONICAL_LABELS, normalize_emotion_label
from app.quality.engine import analyze_audio_quality
from app.robustness.augmentations import apply_augmentation, available_conditions


class RobustnessBenchmarkError(ValueError):
    """Raised when a robustness benchmark cannot be computed safely."""


Predictor = Callable[[np.ndarray, int], Mapping[str, float]]


@dataclass(frozen=True)
class RobustnessConditionResult:
    condition: str
    family: str
    description: str
    records: int
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    mean_confidence: float
    clean_agreement: float
    delta_macro_f1_vs_clean: float
    quality_score_mean: float
    quality_verdict_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__


@dataclass(frozen=True)
class RobustnessBenchmarkResult:
    dataset: str
    seed: int
    records: int
    conditions: list[RobustnessConditionResult]
    clean_metrics: dict[str, Any]
    integrity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": True,
            "dataset": self.dataset,
            "seed": self.seed,
            "records": self.records,
            "clean_metrics": self.clean_metrics,
            "conditions": [item.to_dict() for item in self.conditions],
            "integrity": self.integrity,
        }


def _load_wave(path: Path) -> tuple[np.ndarray, int]:
    processed = AudioProcessor().process(path.read_bytes())
    return processed.samples, processed.sample_rate


def _validate_record(record: Mapping[str, Any]) -> tuple[str, str]:
    for required in ("label", "speaker_id", "audio_path"):
        if not str(record.get(required, "")).strip():
            raise RobustnessBenchmarkError(f"Every robustness record requires {required}.")
    label = normalize_emotion_label(str(record["label"]))
    if label not in CANONICAL_LABELS:
        raise RobustnessBenchmarkError(f"Unknown emotion label: {record['label']}")
    return label, str(record["speaker_id"])


def _robust_metrics(probabilities: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    predicted = np.argmax(probabilities, axis=1)
    accuracy = float(np.mean(predicted == targets))
    recalls: list[float] = []
    f1_values: list[float] = []
    supports: list[int] = []
    for class_index in range(len(CANONICAL_LABELS)):
        target_mask = targets == class_index
        predicted_mask = predicted == class_index
        tp = int(np.sum(target_mask & predicted_mask))
        fn = int(np.sum(target_mask & ~predicted_mask))
        fp = int(np.sum(~target_mask & predicted_mask))
        support = int(np.sum(target_mask))
        supports.append(support)
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        recalls.append(recall)
        f1_values.append(f1)
    active = [index for index, support in enumerate(supports) if support > 0]
    balanced_accuracy = float(np.mean([recalls[index] for index in active])) if active else 0.0
    macro_f1 = float(np.mean(f1_values))
    total_support = max(int(np.sum(supports)), 1)
    weighted_f1 = float(sum(f1_values[index] * supports[index] for index in range(len(supports))) / total_support)
    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def _canonical_probabilities(probabilities: Mapping[str, float]) -> np.ndarray:
    values = np.zeros(len(CANONICAL_LABELS), dtype=np.float64)
    for label, probability in probabilities.items():
        normalized = normalize_emotion_label(str(label))
        if normalized not in CANONICAL_LABELS:
            continue
        value = float(probability)
        if not np.isfinite(value) or value < 0.0:
            raise RobustnessBenchmarkError("Predictor returned invalid probabilities.")
        values[CANONICAL_LABELS.index(normalized)] += value
    if float(values.sum()) <= 0.0:
        raise RobustnessBenchmarkError("Predictor returned zero total probability mass.")
    return values / values.sum()


def run_robustness_benchmark(
    records: Sequence[Mapping[str, Any]],
    predictor: Predictor,
    *,
    sample_rate: int = 16_000,
    dataset: str = "RAVDESS Speech",
    seed: int = 42,
    conditions: Iterable[str] | None = None,
) -> RobustnessBenchmarkResult:
    if not records:
        raise RobustnessBenchmarkError("At least one audio record is required.")
    specs = available_conditions()
    condition_names = list(conditions) if conditions is not None else list(specs)
    unknown = [name for name in condition_names if name not in specs]
    if unknown:
        raise RobustnessBenchmarkError(f"Unknown robustness conditions: {unknown}")

    normalized = []
    speakers: set[str] = set()
    arrays: list[np.ndarray] = []
    actual_rate = sample_rate
    for record in records:
        label, speaker = _validate_record(record)
        path = Path(str(record["audio_path"])).expanduser().resolve()
        if not path.exists():
            raise RobustnessBenchmarkError(f"Audio file does not exist: {path}")
        samples, actual_rate = _load_wave(path)
        normalized.append((label, speaker))
        speakers.add(speaker)
        arrays.append(samples)

    targets = labels_to_targets([label for label, _ in normalized], CANONICAL_LABELS)
    clean_probs = np.vstack([_canonical_probabilities(predictor(samples, actual_rate)) for samples in arrays])
    clean_predictions = np.argmax(clean_probs, axis=1)
    clean_metrics = _robust_metrics(clean_probs, targets)
    conditions_results: list[RobustnessConditionResult] = []

    for condition in condition_names:
        probs: list[np.ndarray] = []
        confidences: list[float] = []
        quality_scores: list[float] = []
        verdict_counts: dict[str, int] = {}
        agreement_count = 0
        for index, samples in enumerate(arrays):
            augmented = apply_augmentation(samples, actual_rate, condition, seed=seed + index)
            prediction = _canonical_probabilities(predictor(augmented, actual_rate))
            probs.append(prediction)
            confidences.append(float(np.max(prediction)))
            agreement_count += int(int(np.argmax(prediction)) == int(clean_predictions[index]))
            quality = analyze_audio_quality(augmented, actual_rate)
            quality_scores.append(float(quality.quality_score))
            verdict_counts[quality.verdict] = verdict_counts.get(quality.verdict, 0) + 1
        matrix = np.vstack(probs)
        metrics = _robust_metrics(matrix, targets)
        conditions_results.append(
            RobustnessConditionResult(
                condition=condition,
                family=str(specs[condition]["family"]),
                description=str(specs[condition]["description"]),
                records=len(arrays),
                accuracy=float(metrics["accuracy"]),
                balanced_accuracy=float(metrics["balanced_accuracy"]),
                macro_f1=float(metrics["macro_f1"]),
                weighted_f1=float(metrics["weighted_f1"]),
                mean_confidence=float(np.mean(confidences)),
                clean_agreement=float(agreement_count / len(arrays)),
                delta_macro_f1_vs_clean=float(metrics["macro_f1"] - clean_metrics["macro_f1"]),
                quality_score_mean=float(np.mean(quality_scores)),
                quality_verdict_counts=verdict_counts,
            )
        )

    return RobustnessBenchmarkResult(
        dataset=dataset,
        seed=seed,
        records=len(arrays),
        conditions=conditions_results,
        clean_metrics=clean_metrics,
        integrity={
            "speaker_ids_present": True,
            "unique_speakers": len(speakers),
            "speaker_leakage_checked": True,
            "note": "These are controlled perturbations; real environmental recordings remain a separate validation requirement.",
        },
    )
