from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.longitudinal_emotion.service import LongitudinalEmotionError, LongitudinalEmotionService


def session_payload(index: int, *, emotion="neutral", valence=0.5, arousal=0.5, dominance=0.5):
    return {
        "session_id": f"s{index}",
        "recorded_at": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
        "duration_seconds": 5.0 + index,
        "emotion": emotion,
        "confidence": 0.7,
        "quality_score": 0.9,
        "baseline_deviation": 0.1 * min(index, 5),
        "acoustic": {
            "pitch_mean_hz": 180 + index * 15,
            "pitch_std_hz": 20 + index,
            "energy_mean": 0.05 + index * 0.01,
            "estimated_syllable_rate_sps": 2.0 + index * 0.1,
            "estimated_pause_ratio": max(0.05 - index * 0.005, 0.01),
        },
        "affect": {"valence": valence, "arousal": arousal, "dominance": dominance},
    }


def test_empty_history_is_descriptive(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    sessions, trends = service.get("browser-1")
    assert sessions == []
    assert trends["session_count"] == 0
    assert "No voice-history" in trends["summary"]


def test_append_sorts_and_replaces_same_session(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    service.append("browser-1", session_payload(2))
    service.append("browser-1", session_payload(1, emotion="happy"))
    service.append("browser-1", session_payload(2, emotion="sad"))
    sessions, trends = service.get("browser-1")
    assert [item.session_id for item in sessions] == ["s1", "s2"]
    assert sessions[-1].emotion == "sad"
    assert trends["latest_emotion"] == "sad"


def test_affect_trend_detects_direction(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    for i in range(5):
        service.append("browser-1", session_payload(i, valence=0.2 + i * 0.15, arousal=0.4, dominance=0.5))
    _, trends = service.get("browser-1")
    valence = next(item for item in trends["affect_trends"] if item["metric"] == "Valence")
    assert valence["direction"] == "increasing"
    assert valence["delta_early_to_recent"] > 0
    assert valence["strength"] > 0


def test_acoustic_trend_detects_decrease(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    for i in range(5):
        payload = session_payload(i)
        payload["acoustic"]["estimated_pause_ratio"] = 0.45 - i * 0.07
        service.append("browser-1", payload)
    _, trends = service.get("browser-1")
    pause = next(item for item in trends["acoustic_trends"] if item["metric"] == "Pause ratio")
    assert pause["direction"] == "decreasing"


def test_insufficient_data_is_explicit(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    service.append("browser-1", session_payload(0))
    _, trends = service.get("browser-1")
    assert all(item["direction"] == "insufficient_data" for item in trends["affect_trends"])
    assert "More saved sessions" in trends["summary"]


def test_history_is_capped(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    for i in range(105):
        service.append("browser-1", session_payload(i))
    sessions, trends = service.get("browser-1")
    assert len(sessions) == 100
    assert trends["session_count"] == 100
    assert sessions[0].session_id == "s5"


def test_invalid_profile_id_rejected(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    try:
        service.get("../escape")
    except LongitudinalEmotionError as exc:
        assert "Profile ID" in str(exc)
    else:
        raise AssertionError("Expected invalid profile ID to be rejected")


def test_invalid_numeric_value_rejected(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    payload = session_payload(0)
    payload["affect"]["valence"] = 2.0
    try:
        service.append("browser-1", payload)
    except LongitudinalEmotionError:
        pass
    else:
        raise AssertionError("Expected out-of-range affect to be rejected")


def test_vocal_consistency_changes_with_cross_session_variability(tmp_path):
    stable_service = LongitudinalEmotionService(tmp_path / "stable")
    for i in range(5):
        payload = session_payload(i)
        payload["acoustic"]["pitch_mean_hz"] = 200
        payload["acoustic"]["energy_mean"] = 0.05
        stable_service.append("browser-1", payload)
    _, stable_trends = stable_service.get("browser-1")

    variable_service = LongitudinalEmotionService(tmp_path / "variable")
    for i in range(5):
        payload = session_payload(i)
        payload["acoustic"]["pitch_mean_hz"] = 120 + i * 40
        payload["acoustic"]["energy_mean"] = 0.02 + i * 0.03
        variable_service.append("browser-1", payload)
    _, variable_trends = variable_service.get("browser-1")

    assert stable_trends["vocal_consistency"] > variable_trends["vocal_consistency"]
