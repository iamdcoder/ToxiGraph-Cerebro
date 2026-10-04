from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np


class PersonalBaselineError(ValueError):
    """Raised when a personal acoustic baseline cannot be created or used."""


PROFILE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_FILE_LOCK = Lock()
PROFILE_VERSION = "personal-baseline-v1"
MIN_SAMPLES = 3
MAX_SAMPLES = 8


METRIC_SPECS: tuple[tuple[str, str, float], ...] = (
    ("pitch_mean_hz", "pitch_hz.mean", 5.0),
    ("pitch_std_hz", "pitch_hz.std", 2.0),
    ("energy_mean", "energy.mean", 0.005),
    ("energy_std", "energy.std", 0.002),
    ("spectral_centroid_mean_hz", "spectral_centroid_hz.mean", 50.0),
    ("spectral_centroid_std_hz", "spectral_centroid_hz.std", 30.0),
    ("spectral_bandwidth_mean_hz", "spectral_bandwidth_hz.mean", 50.0),
    ("spectral_bandwidth_std_hz", "spectral_bandwidth_hz.std", 20.0),
    ("spectral_rolloff_mean_hz", "spectral_rolloff_hz.mean", 80.0),
    ("zero_crossing_rate_mean", "zero_crossing_rate.mean", 0.01),
    ("estimated_syllable_rate_sps", "estimated_syllable_rate_sps", 0.20),
    ("estimated_pause_ratio", "estimated_pause_ratio", 0.03),
)


@dataclass(frozen=True)
class BaselineMetric:
    center: float
    scale: float
    observed_min: float
    observed_max: float
    sample_spread: float
    unit: str = ""

    def to_dict(self) -> dict[str, float | str]:
        return {
            "center": round(self.center, 6),
            "scale": round(self.scale, 6),
            "observed_min": round(self.observed_min, 6),
            "observed_max": round(self.observed_max, 6),
            "sample_spread": round(self.sample_spread, 6),
            "unit": self.unit,
        }


@dataclass(frozen=True)
class PersonalBaselineProfile:
    profile_id: str
    version: str
    created_at: str
    updated_at: str
    sample_count: int
    reference_duration_seconds: float
    metrics: dict[str, BaselineMetric] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sample_count": self.sample_count,
            "reference_duration_seconds": round(self.reference_duration_seconds, 4),
            "metrics": {key: value.to_dict() for key, value in self.metrics.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PersonalBaselineProfile":
        try:
            metrics = {
                key: BaselineMetric(
                    center=float(value["center"]),
                    scale=float(value["scale"]),
                    observed_min=float(value["observed_min"]),
                    observed_max=float(value["observed_max"]),
                    sample_spread=float(value["sample_spread"]),
                    unit=str(value.get("unit", "")),
                )
                for key, value in payload["metrics"].items()
            }
            return cls(
                profile_id=str(payload["profile_id"]),
                version=str(payload["version"]),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                sample_count=int(payload["sample_count"]),
                reference_duration_seconds=float(payload["reference_duration_seconds"]),
                metrics=metrics,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PersonalBaselineError("The stored personal baseline is invalid or corrupted.") from exc


@dataclass(frozen=True)
class BaselineMetricComparison:
    name: str
    current_value: float
    baseline_value: float
    scale: float
    robust_z: float
    deviation_ratio: float
    direction: str
    significance: str
    unit: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "current_value": round(self.current_value, 6),
            "baseline_value": round(self.baseline_value, 6),
            "scale": round(self.scale, 6),
            "robust_z": round(self.robust_z, 3),
            "deviation_ratio": round(self.deviation_ratio, 3),
            "direction": self.direction,
            "significance": self.significance,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class PersonalBaselineComparison:
    profile_id: str
    profile_version: str
    sample_count: int
    overall_deviation: float
    normality: str
    summary: str
    metrics: list[BaselineMetricComparison]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "sample_count": self.sample_count,
            "overall_deviation": round(self.overall_deviation, 3),
            "normality": self.normality,
            "summary": self.summary,
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


class PersonalBaselineService:
    def __init__(self, root_path: str | os.PathLike[str] = "artifacts/personal_baselines") -> None:
        self.root = Path(root_path)
        self.root.mkdir(parents=True, exist_ok=True)

    def _validate_profile_id(self, profile_id: str) -> str:
        value = profile_id.strip()
        if not PROFILE_ID_PATTERN.fullmatch(value):
            raise PersonalBaselineError(
                "Profile ID must be 1–64 characters using letters, numbers, '-' or '_'."
            )
        return value

    def _path(self, profile_id: str) -> Path:
        return self.root / f"{profile_id}.json"

    def exists(self, profile_id: str) -> bool:
        return self._path(self._validate_profile_id(profile_id)).exists()

    def delete(self, profile_id: str) -> None:
        path = self._path(self._validate_profile_id(profile_id))
        with _FILE_LOCK:
            if not path.exists():
                raise PersonalBaselineError("Personal baseline profile was not found.")
            path.unlink()

    def load(self, profile_id: str) -> PersonalBaselineProfile:
        path = self._path(self._validate_profile_id(profile_id))
        if not path.exists():
            raise PersonalBaselineError("Personal baseline profile was not found.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PersonalBaselineError("The personal baseline profile could not be read.") from exc
        return PersonalBaselineProfile.from_dict(payload)

    def create(self, profile_id: str, feature_payloads: list[dict[str, Any]]) -> PersonalBaselineProfile:
        profile_id = self._validate_profile_id(profile_id)
        if not MIN_SAMPLES <= len(feature_payloads) <= MAX_SAMPLES:
            raise PersonalBaselineError(
                f"A personal baseline needs {MIN_SAMPLES}–{MAX_SAMPLES} reference samples."
            )

        extracted: dict[str, list[float]] = {name: [] for name, _, _ in METRIC_SPECS}
        durations: list[float] = []
        for payload in feature_payloads:
            duration = float(payload.get("duration_seconds", 0.0))
            if duration <= 0:
                raise PersonalBaselineError("Every baseline sample must contain a valid duration.")
            durations.append(duration)
            for metric_name, path, _ in METRIC_SPECS:
                value = _read_nested(payload, path)
                if not math.isfinite(value):
                    raise PersonalBaselineError(f"Baseline feature '{metric_name}' is not finite.")
                extracted[metric_name].append(value)

        now = datetime.now(timezone.utc).isoformat()
        old_profile: PersonalBaselineProfile | None = None
        path = self._path(profile_id)
        if path.exists():
            old_profile = self.load(profile_id)

        metrics: dict[str, BaselineMetric] = {}
        for metric_name, _, floor in METRIC_SPECS:
            values = np.asarray(extracted[metric_name], dtype=np.float64)
            center = float(np.median(values))
            mad = float(np.median(np.abs(values - center)))
            std = float(np.std(values))
            scale = max(1.4826 * mad, std, floor)
            metrics[metric_name] = BaselineMetric(
                center=center,
                scale=scale,
                observed_min=float(np.min(values)),
                observed_max=float(np.max(values)),
                sample_spread=std,
                unit=_unit_for_metric(metric_name),
            )

        profile = PersonalBaselineProfile(
            profile_id=profile_id,
            version=PROFILE_VERSION,
            created_at=old_profile.created_at if old_profile else now,
            updated_at=now,
            sample_count=len(feature_payloads),
            reference_duration_seconds=float(np.mean(durations)),
            metrics=metrics,
        )
        self._atomic_write(path, profile.to_dict())
        return profile

    def compare(self, profile_id: str, features: dict[str, Any]) -> PersonalBaselineComparison:
        profile = self.load(profile_id)
        comparisons: list[BaselineMetricComparison] = []

        for metric_name, path, _ in METRIC_SPECS:
            baseline = profile.metrics.get(metric_name)
            if baseline is None:
                continue
            current = _read_nested(features, path)
            delta = current - baseline.center
            robust_z = abs(delta) / max(baseline.scale, 1e-9)
            ratio = abs(delta) / max(abs(baseline.center), baseline.scale, 1e-9)
            direction = "above" if delta > baseline.scale * 0.15 else "below" if delta < -baseline.scale * 0.15 else "within"
            significance = _significance(robust_z)
            comparisons.append(
                BaselineMetricComparison(
                    name=metric_name,
                    current_value=current,
                    baseline_value=baseline.center,
                    scale=baseline.scale,
                    robust_z=robust_z,
                    deviation_ratio=ratio,
                    direction=direction,
                    significance=significance,
                    unit=baseline.unit,
                )
            )

        if not comparisons:
            raise PersonalBaselineError("The personal baseline contains no comparable acoustic metrics.")

        bounded = [min(item.robust_z / 3.0, 1.0) for item in comparisons]
        overall_mean = float(np.mean(bounded))
        overall_peak = float(np.max(bounded))
        overall = float(0.55 * overall_mean + 0.45 * overall_peak)
        significant = [item for item in comparisons if item.significance in {"notable", "strong"}]
        if overall < 0.20:
            normality = "typical"
        elif overall < 0.45:
            normality = "slightly_deviant"
        else:
            normality = "strongly_deviant"

        summary = _build_summary(normality, significant)
        return PersonalBaselineComparison(
            profile_id=profile.profile_id,
            profile_version=profile.version,
            sample_count=profile.sample_count,
            overall_deviation=overall,
            normality=normality,
            summary=summary,
            metrics=sorted(comparisons, key=lambda item: item.robust_z, reverse=True),
        )

    def _atomic_write(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, indent=2, sort_keys=True)
        with _FILE_LOCK:
            fd, temp_path = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)


def _read_nested(payload: dict[str, Any], path: str) -> float:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise PersonalBaselineError(f"Missing acoustic baseline feature: {path}.")
        current = current[part]
    try:
        return float(current)
    except (TypeError, ValueError) as exc:
        raise PersonalBaselineError(f"Acoustic baseline feature '{path}' is not numeric.") from exc


def _unit_for_metric(metric_name: str) -> str:
    if metric_name.endswith("_hz"):
        return "Hz"
    if metric_name.endswith("_sps"):
        return "per_second"
    if metric_name.endswith("ratio"):
        return "ratio"
    return "normalized"


def _significance(robust_z: float) -> str:
    if robust_z >= 2.5:
        return "strong"
    if robust_z >= 1.5:
        return "notable"
    if robust_z >= 0.75:
        return "mild"
    return "within_baseline"


def _build_summary(normality: str, significant: list[BaselineMetricComparison]) -> str:
    if normality == "typical":
        return "The current acoustic profile is close to this speaker's personal baseline."
    if not significant:
        return "The current acoustic profile is somewhat different from baseline, but no single measured feature is strongly outside the reference range."
    names = ", ".join(_human_metric(item.name) for item in significant[:3])
    descriptor = "strongly" if normality == "strongly_deviant" else "noticeably"
    return f"The current acoustic profile is {descriptor} different from baseline, led by {names}."


def _human_metric(name: str) -> str:
    replacements = {
        "pitch_mean_hz": "mean pitch",
        "pitch_std_hz": "pitch variation",
        "energy_mean": "vocal energy",
        "energy_std": "energy variation",
        "spectral_centroid_mean_hz": "spectral centroid",
        "spectral_centroid_std_hz": "spectral centroid variation",
        "spectral_bandwidth_mean_hz": "spectral bandwidth",
        "spectral_bandwidth_std_hz": "bandwidth variation",
        "spectral_rolloff_mean_hz": "spectral rolloff",
        "zero_crossing_rate_mean": "zero-crossing rate",
        "estimated_syllable_rate_sps": "speech rate",
        "estimated_pause_ratio": "pause ratio",
    }
    return replacements.get(name, name.replace("_", " "))
