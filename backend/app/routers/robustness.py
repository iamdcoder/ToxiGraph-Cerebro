from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.robustness import RobustnessReportResponse, RobustnessStatusResponse

router = APIRouter()


def _report_path() -> Path:
    return Path(settings.ROBUSTNESS_REPORT_PATH).expanduser().resolve()


def _load_report() -> dict:
    path = _report_path()
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Robustness report is unavailable or invalid.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Robustness report must be a JSON object.")
    return payload


@router.get("/status", response_model=RobustnessStatusResponse)
def robustness_status() -> RobustnessStatusResponse:
    path = _report_path()
    if not path.exists():
        return RobustnessStatusResponse(
            available=False,
            message="No real robustness report has been generated yet.",
        )
    try:
        payload = _load_report()
    except ValueError as exc:
        return RobustnessStatusResponse(available=False, message=str(exc))
    return RobustnessStatusResponse(
        available=True,
        dataset=payload.get("dataset"),
        records=int(payload.get("records", 0)),
        conditions=len(payload.get("conditions", [])),
        message="Real robustness report is available.",
    )


@router.get("/report", response_model=RobustnessReportResponse)
def robustness_report() -> RobustnessReportResponse:
    try:
        return RobustnessReportResponse(**_load_report())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="No robustness report has been generated yet.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
