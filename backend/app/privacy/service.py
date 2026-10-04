from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

from app.security import SecurityPolicyError, validate_profile_id


@dataclass(frozen=True)
class PrivacyArtifact:
    category: str
    path: Path


class PrivacyService:
    """Centralized profile-data inventory, retention and deletion service."""

    CATEGORIES = (
        "personal_baseline",
        "voice_history",
        "personalization",
    )

    def __init__(
        self,
        *,
        personal_baseline_dir: str | os.PathLike[str],
        voice_history_dir: str | os.PathLike[str],
        personalization_dir: str | os.PathLike[str],
        retention_days: int = 30,
        retention_enabled: bool = True,
    ) -> None:
        if retention_days <= 0:
            raise ValueError("Retention days must be positive.")
        self.roots = {
            "personal_baseline": Path(personal_baseline_dir),
            "voice_history": Path(voice_history_dir),
            "personalization": Path(personalization_dir),
        }
        self.retention_days = int(retention_days)
        self.retention_enabled = bool(retention_enabled)
        self._lock = Lock()

    def status(self, profile_id: str) -> dict:
        profile_id = validate_profile_id(profile_id)
        artifacts = self._inventory(profile_id)
        return {
            "profile_id": profile_id,
            "retention_enabled": self.retention_enabled,
            "retention_days": self.retention_days,
            "raw_audio_stored": False,
            "raw_transcripts_stored": False,
            "categories": {
                category: bool(any(item.category == category for item in artifacts))
                for category in self.CATEGORIES
            },
            "stored_file_count": len(artifacts),
            "stored_categories": sorted({item.category for item in artifacts}),
            "message": (
                "CEREBRO stores compact profile summaries and explicit calibration feedback; raw microphone recordings are not persisted by these services."
            ),
        }

    def delete(self, profile_id: str) -> dict:
        profile_id = validate_profile_id(profile_id)
        deleted: list[str] = []
        with self._lock:
            for artifact in self._inventory(profile_id):
                try:
                    artifact.path.unlink()
                    deleted.append(artifact.category)
                except FileNotFoundError:
                    continue
        return {
            "profile_id": profile_id,
            "deleted_categories": sorted(set(deleted)),
            "deleted_file_count": len(deleted),
            "success": True,
            "message": "All stored CEREBRO profile data for this browser profile has been deleted.",
        }

    def prune_expired(self) -> dict:
        if not self.retention_enabled:
            return {"enabled": False, "deleted_file_count": 0, "message": "Automatic retention pruning is disabled."}
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.retention_days)
        deleted = 0
        with self._lock:
            for root in self.roots.values():
                if not root.exists():
                    continue
                for path in root.glob("*.json"):
                    try:
                        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
                    except OSError:
                        continue
                    if modified < cutoff:
                        try:
                            path.unlink()
                            deleted += 1
                        except OSError:
                            continue
        return {
            "enabled": True,
            "retention_days": self.retention_days,
            "deleted_file_count": deleted,
            "message": "Expired profile data was pruned." if deleted else "No expired profile data required pruning.",
        }

    def _inventory(self, profile_id: str) -> list[PrivacyArtifact]:
        validate_profile_id(profile_id)
        output: list[PrivacyArtifact] = []
        for category, root in self.roots.items():
            path = root / f"{profile_id}.json"
            if path.exists() and path.is_file():
                output.append(PrivacyArtifact(category, path))
        return output
