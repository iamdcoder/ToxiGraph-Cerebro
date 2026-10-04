from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_optimization_status_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "OPTIMIZATION_REPORT_PATH", str(tmp_path / "missing.json"))
    payload = TestClient(app).get("/api/v1/evaluation/optimization/status")
    assert payload.status_code == 200
    body = payload.json()
    assert body["available"] is False
    assert body["selected_model"] is None


def test_optimization_status_and_report(tmp_path, monkeypatch):
    report = {
        "success": True,
        "selected_model": "linear_svm",
        "cv": {"folds": 4},
        "test": {"macro_f1": 0.71, "accuracy": 0.72},
        "candidate_results": {},
    }
    path = tmp_path / "optimization.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(settings, "OPTIMIZATION_REPORT_PATH", str(path))
    client = TestClient(app)
    status = client.get("/api/v1/evaluation/optimization/status")
    assert status.status_code == 200
    assert status.json()["available"] is True
    assert status.json()["selected_model"] == "linear_svm"
    response = client.get("/api/v1/evaluation/optimization/report")
    assert response.status_code == 200
    assert response.json()["test"]["macro_f1"] == 0.71
