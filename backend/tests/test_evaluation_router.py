from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app

client = TestClient(app)


def test_evaluation_status_reports_unavailable_without_report(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_PATH", str(tmp_path / "missing.json"))
    response = client.get("/api/v1/evaluation/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is False
    assert payload["test_records"] == 0


def test_evaluation_status_and_report_expose_real_artifact(tmp_path, monkeypatch):
    report_path = tmp_path / "evaluation.json"
    report_path.write_text(
        json.dumps(
            {
                "success": True,
                "dataset": "fixture",
                "test_records": 14,
                "test_speakers": ["test-0", "test-1"],
                "comparison": {"classical_raw": {"accuracy": 0.5}},
                "integrity": {"speaker_leakage": False},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "EVALUATION_REPORT_PATH", str(report_path))
    status = client.get("/api/v1/evaluation/status")
    assert status.status_code == 200
    assert status.json()["available"] is True
    assert status.json()["models"] == ["classical_raw"]

    report = client.get("/api/v1/evaluation/report")
    assert report.status_code == 200
    assert report.json()["dataset"] == "fixture"


def test_evaluation_report_returns_404_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "EVALUATION_REPORT_PATH", str(tmp_path / "missing.json"))
    response = client.get("/api/v1/evaluation/report")
    assert response.status_code == 404
