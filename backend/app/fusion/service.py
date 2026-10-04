from __future__ import annotations

from threading import Lock

from app.config import settings
from app.fusion.engine import FusionEngine
from app.fusion.training import FusionTrainingError, load_artifact

_artifact = None
_engine = None
_lock = Lock()


def get_engine() -> FusionEngine:
    global _artifact, _engine
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is None:
            _artifact = load_artifact(settings.FUSION_CALIBRATION_PATH)
            _engine = FusionEngine(_artifact)
    return _engine


def clear_engine_cache() -> None:
    global _artifact, _engine
    with _lock:
        _artifact = None
        _engine = None


def get_calibration_status() -> dict:
    try:
        engine = get_engine()
    except FusionTrainingError as exc:
        return {
            "available": False,
            "message": str(exc),
        }
    return {
        "available": True,
        "version": engine.artifact.version,
        "labels": list(engine.artifact.labels),
        "classical_temperature": engine.artifact.classical_temperature,
        "deep_temperature": engine.artifact.deep_temperature,
        "classical_weight": engine.artifact.classical_weight,
        "deep_weight": engine.artifact.deep_weight,
        "validation_metrics": engine.artifact.validation_metrics,
        "metadata": engine.artifact.metadata,
    }
