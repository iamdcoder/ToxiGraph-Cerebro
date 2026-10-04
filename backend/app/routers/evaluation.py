from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings

router = APIRouter()


def _report_path() -> Path:
    return Path(settings.EVALUATION_REPORT_PATH).expanduser().resolve()


def _load_report() -> dict:
    path = _report_path()
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Evaluation report is unavailable or invalid.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Evaluation report must be a JSON object.")
    return payload


@router.get("/status")
def evaluation_status() -> dict:
    path = _report_path()
    if not path.exists():
        return {
            "available": False,
            "dataset": None,
            "test_records": 0,
            "test_speakers": [],
            "models": [],
            "message": "No real held-out evaluation report has been generated yet.",
        }
    try:
        payload = _load_report()
    except ValueError as exc:
        return {
            "available": False,
            "dataset": None,
            "test_records": 0,
            "test_speakers": [],
            "models": [],
            "message": str(exc),
        }
    return {
        "available": True,
        "dataset": payload.get("dataset"),
        "test_records": payload.get("test_records", 0),
        "test_speakers": payload.get("test_speakers", []),
        "models": sorted(payload.get("comparison", {}).keys()),
        "integrity": payload.get("integrity", {}),
        "message": "Real held-out evaluation report is available.",
    }


@router.get("/report")
def evaluation_report() -> dict:
    try:
        return _load_report()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No evaluation report has been generated yet.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
