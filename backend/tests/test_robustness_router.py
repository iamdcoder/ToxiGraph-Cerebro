from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app


def test_robustness_status_is_safe_when_report_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "ROBUSTNESS_REPORT_PATH", str(tmp_path / "missing.json"))
    response = TestClient(app).get("/api/v1/robustness/status")
    assert response.status_code == 200
    assert response.json()["available"] is False


def test_robustness_report_returns_404_when_report_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(settings, "ROBUSTNESS_REPORT_PATH", str(tmp_path / "missing.json"))
    response = TestClient(app).get("/api/v1/robustness/report")
    assert response.status_code == 404


def test_robustness_report_reads_real_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = tmp_path / "robustness.json"
    report.write_text(
        '{"success": true, "dataset": "test", "seed": 42, "records": 2, "clean_metrics": {"macro_f1": 0.8}, "conditions": [], "integrity": {"speaker_leakage_checked": true}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(settings, "ROBUSTNESS_REPORT_PATH", str(report))
    response = TestClient(app).get("/api/v1/robustness/report")
    assert response.status_code == 200
    assert response.json()["dataset"] == "test"
