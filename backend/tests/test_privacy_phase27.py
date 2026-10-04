from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

import app.routers.privacy as privacy_router
from app.main import app
from app.privacy.service import PrivacyService


def make_service(tmp_path: Path, retention_days: int = 30) -> PrivacyService:
    return PrivacyService(
        personal_baseline_dir=tmp_path / "baseline",
        voice_history_dir=tmp_path / "history",
        personalization_dir=tmp_path / "personalization",
        retention_days=retention_days,
        retention_enabled=True,
    )


def test_privacy_status_starts_empty(tmp_path):
    service = make_service(tmp_path)
    payload = service.status("browser-1")
    assert payload["raw_audio_stored"] is False
    assert payload["raw_transcripts_stored"] is False
    assert payload["stored_file_count"] == 0


def test_privacy_status_inventory_and_delete(tmp_path):
    service = make_service(tmp_path)
    for category, root in service.roots.items():
        root.mkdir(parents=True, exist_ok=True)
        (root / "browser-1.json").write_text('{"example":true}', encoding="utf-8")
    before = service.status("browser-1")
    assert before["stored_file_count"] == 3
    assert set(before["stored_categories"]) == set(service.CATEGORIES)

    deleted = service.delete("browser-1")
    assert deleted["success"] is True
    assert deleted["deleted_file_count"] == 3
    assert service.status("browser-1")["stored_file_count"] == 0


def test_privacy_retention_prunes_expired_profile_files(tmp_path):
    service = make_service(tmp_path, retention_days=30)
    root = service.roots["voice_history"]
    root.mkdir(parents=True, exist_ok=True)
    old_file = root / "old.json"
    recent_file = root / "recent.json"
    old_file.write_text("{}", encoding="utf-8")
    recent_file.write_text("{}", encoding="utf-8")
    old_timestamp = (datetime.now(timezone.utc) - timedelta(days=45)).timestamp()
    os.utime(old_file, (old_timestamp, old_timestamp))

    report = service.prune_expired()
    assert report["deleted_file_count"] == 1
    assert not old_file.exists()
    assert recent_file.exists()


def test_privacy_service_rejects_unsafe_profile_id(tmp_path):
    service = make_service(tmp_path)
    try:
        service.status("../secret")
    except ValueError as exc:
        assert "Profile ID" in str(exc)
    else:
        raise AssertionError("Expected unsafe profile ID to be rejected")


def test_privacy_status_route(monkeypatch, tmp_path):
    service = make_service(tmp_path)
    monkeypatch.setattr(privacy_router, "get_service", lambda: service)
    with TestClient(app) as client:
        response = client.get("/api/v1/privacy/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["raw_audio_stored"] is False
    assert payload["retention_days"] >= 1


def test_privacy_profile_route_reports_categories(monkeypatch, tmp_path):
    service = make_service(tmp_path)
    root = service.roots["voice_history"]
    root.mkdir(parents=True, exist_ok=True)
    (root / "browser-1.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(privacy_router, "get_service", lambda: service)
    with TestClient(app) as client:
        response = client.get("/api/v1/privacy/browser-1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["stored_file_count"] == 1
    assert payload["categories"]["voice_history"] is True


def test_privacy_delete_route_deletes_all_categories(monkeypatch, tmp_path):
    service = make_service(tmp_path)
    for root in service.roots.values():
        root.mkdir(parents=True, exist_ok=True)
        (root / "browser-1.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(privacy_router, "get_service", lambda: service)
    with TestClient(app) as client:
        response = client.delete("/api/v1/privacy/browser-1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["success"] is True
    assert payload["deleted_file_count"] == 3
    assert service.status("browser-1")["stored_file_count"] == 0
