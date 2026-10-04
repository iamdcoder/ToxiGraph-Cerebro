from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings

router = APIRouter()


def _report_path() -> Path:
    return Path(settings.OPTIMIZATION_REPORT_PATH).expanduser().resolve()


def _load_report() -> dict:
    path = _report_path()
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Optimization report is unavailable or invalid.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Optimization report must be a JSON object.")
    return payload


@router.get("/status")
def optimization_status() -> dict:
    path = _report_path()
    if not path.exists():
        return {
            "available": False,
            "selected_model": None,
            "message": "No real-data optimization report has been generated yet.",
        }
    try:
        payload = _load_report()
    except ValueError as exc:
        return {"available": False, "selected_model": None, "message": str(exc)}
    return {
        "available": True,
        "selected_model": payload.get("selected_model"),
        "test": payload.get("test", {}),
        "cv": payload.get("cv", {}),
        "message": "Real-data optimization report is available.",
    }


@router.get("/report")
def optimization_report() -> dict:
    try:
        return _load_report()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No optimization report has been generated yet.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
