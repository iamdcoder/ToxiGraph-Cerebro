from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.evaluation.benchmark import benchmark_prediction_files
from app.evaluation.html_report import render_benchmark_html
from app.evaluation.metrics import (
    EvaluationError,
    calibration_curve,
    comprehensive_metrics,
    metric_deltas,
    speaker_level_summary,
)
from app.fusion.constants import CANONICAL_LABELS
from app.fusion.training import fit_fusion_calibrator, save_artifact


def sharp(label: str, high: float = 0.8) -> dict[str, float]:
    rest = (1.0 - high) / (len(CANONICAL_LABELS) - 1)
    return {emotion: (high if emotion == label else rest) for emotion in CANONICAL_LABELS}


def make_rows(count: int = 42) -> list[dict]:
    rows = []
    for index in range(count):
        label = CANONICAL_LABELS[index % len(CANONICAL_LABELS)]
        classical = label if index % 7 else CANONICAL_LABELS[(index + 1) % len(CANONICAL_LABELS)]
        deep = label if index % 11 else CANONICAL_LABELS[(index + 2) % len(CANONICAL_LABELS)]
        rows.append(
            {
                "id": f"row-{index}",
                "speaker_id": f"spk-{index // 7}",
                "label": label,
                "classical_probabilities": sharp(classical, 0.76),
                "deep_probabilities": sharp(deep, 0.82),
            }
        )
    return rows


def test_comprehensive_metrics_contains_confusion_and_per_class():
    probabilities = np.vstack([[sharp(label, 0.9)[emotion] for emotion in CANONICAL_LABELS] for label in CANONICAL_LABELS for _ in range(2)])
    targets = np.repeat(np.arange(len(CANONICAL_LABELS)), 2)
    report = comprehensive_metrics(probabilities, targets)
    assert report["accuracy"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx(1.0)
    assert len(report["confusion_matrix"]) == len(CANONICAL_LABELS)
    assert set(report["per_class"]) == set(CANONICAL_LABELS)
    assert len(report["calibration_curve"]) == 10


def test_calibration_curve_is_bounded_and_counts_sum():
    rows = np.vstack([[sharp(label, high)[emotion] for emotion in CANONICAL_LABELS] for label, high in (("angry", 0.6), ("sad", 0.9))])
    targets = np.array([CANONICAL_LABELS.index("angry"), CANONICAL_LABELS.index("sad")])
    curve = calibration_curve(rows, targets, bins=5)
    assert sum(item["count"] for item in curve) == 2
    assert all(0 <= item["mean_confidence"] <= 1 for item in curve)


def test_invalid_probability_matrix_is_rejected():
    with pytest.raises(EvaluationError):
        comprehensive_metrics(np.zeros((2, len(CANONICAL_LABELS))), np.array([0, 1]))


def test_metric_deltas_compare_to_reference():
    reports = {
        "classical_raw": {"accuracy": 0.5, "macro_f1": 0.4, "nll": 1.0, "brier": 0.4, "ece": 0.2},
        "fused": {"accuracy": 0.6, "macro_f1": 0.5, "nll": 0.8, "brier": 0.3, "ece": 0.1},
    }
    delta = metric_deltas(reports, "classical_raw")
    assert delta["fused"]["accuracy_delta"] == pytest.approx(0.1)
    assert delta["fused"]["nll_delta"] == pytest.approx(-0.2)


def test_speaker_level_summary_groups_records():
    rows = make_rows(14)
    summary = speaker_level_summary(rows, "classical_probabilities")
    assert set(summary) == {"spk-0", "spk-1"}
    assert sum(int(item["records"]) for item in summary.values()) == 14


def test_benchmark_rejects_speaker_leakage(tmp_path):
    rows = make_rows(42)
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    validation.write_text("\n".join(json.dumps(row) for row in rows[:21]), encoding="utf-8")
    leaking = dict(rows[21])
    leaking["speaker_id"] = rows[0]["speaker_id"]
    test_rows = [leaking, *rows[22:42]]
    test.write_text("\n".join(json.dumps(row) for row in test_rows), encoding="utf-8")
    artifact = fit_fusion_calibrator(rows)
    with pytest.raises(EvaluationError, match="Speaker leakage"):
        benchmark_prediction_files(validation, test, artifact)


def test_benchmark_produces_all_model_reports_and_integrity(tmp_path):
    rows = make_rows(42)
    validation_rows = [dict(row, speaker_id=f"val-{i // 7}") for i, row in enumerate(rows)]
    test_rows = [dict(row, speaker_id=f"test-{i // 7}") for i, row in enumerate(rows)]
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    validation.write_text("\n".join(json.dumps(row) for row in validation_rows), encoding="utf-8")
    test.write_text("\n".join(json.dumps(row) for row in test_rows), encoding="utf-8")
    artifact = fit_fusion_calibrator(validation_rows)
    result = benchmark_prediction_files(validation, test, artifact)
    assert result.integrity["speaker_leakage"] is False
    assert set(result.comparison) == {
        "classical_raw",
        "deep_raw",
        "classical_calibrated",
        "deep_calibrated",
        "fused",
    }
    assert set(result.deltas_vs_classical_raw) == {
        "deep_raw",
        "classical_calibrated",
        "deep_calibrated",
        "fused",
    }
    assert set(result.speaker_level) == set(result.comparison)


def test_html_report_writes_machine_readable_payload(tmp_path):
    destination = tmp_path / "evaluation.html"
    path = render_benchmark_html({"dataset": "fixture", "test_records": 3, "comparison": {}}, destination)
    assert path.exists()
    assert "CEREBRO" in path.read_text(encoding="utf-8")
