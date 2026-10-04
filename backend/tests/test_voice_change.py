from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.longitudinal_emotion.service import LongitudinalEmotionService
from app.voice_change.engine import MIN_HISTORY_SESSIONS, VoiceChangeEngine


def history_session(index: int, *, pitch=200.0, energy=0.05, valence=0.5, arousal=0.5, emotion="neutral"):
    return {
        "session_id": f"s{index}",
        "recorded_at": (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat(),
        "duration_seconds": 5,
        "emotion": emotion,
        "confidence": 0.8,
        "quality_score": 0.9,
        "acoustic": {
            "pitch_mean_hz": pitch,
            "pitch_std_hz": 20,
            "energy_mean": energy,
            "estimated_syllable_rate_sps": 2.5,
            "estimated_pause_ratio": 0.15,
            "spectral_centroid_mean_hz": 1800,
        },
        "affect": {"valence": valence, "arousal": arousal, "dominance": 0.5},
    }


def make_sessions(service: LongitudinalEmotionService, count=5):
    for i in range(count):
        service.append("p1", history_session(i))
    return service.get("p1")[0]


def test_requires_three_prior_sessions(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    for i in range(MIN_HISTORY_SESSIONS):
        service.append("p1", history_session(i))
    sessions = service.get("p1")[0]
    current = sessions[-1]
    report = VoiceChangeEngine().analyze("p1", current, sessions[:-1])
    assert report.available is False
    assert report.level == "insufficient_data"
    assert report.historical_session_count == MIN_HISTORY_SESSIONS - 1


def test_stable_session_is_normal(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    sessions = make_sessions(service, 5)
    current = sessions[-1]
    report = VoiceChangeEngine().analyze("p1", current, sessions[:-1])
    assert report.available is True
    assert report.level == "normal"
    assert report.change_score < 0.25


def test_large_acoustic_shift_is_unusual(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    make_sessions(service, 5)
    current_payload = history_session(8, pitch=360, energy=0.22, valence=0.1, arousal=0.92)
    current = service.append("p1", current_payload)[0]
    prior = service.get("p1")[0][:-1]
    report = VoiceChangeEngine().analyze("p1", current, prior)
    assert report.change_score >= 0.68
    assert report.level == "unusual"
    assert report.strongest_changes
    assert report.strongest_changes[0].robust_z >= 2.5


def test_change_direction_is_exposed(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    make_sessions(service, 5)
    current_payload = history_session(9, pitch=155, energy=0.025, valence=0.2, arousal=0.85)
    current = service.append("p1", current_payload)[0]
    prior = service.get("p1")[0][:-1]
    report = VoiceChangeEngine().analyze("p1", current, prior)
    by_name = {item.name: item for item in report.strongest_changes}
    assert by_name["Mean pitch"].direction == "below"
    assert by_name["Mean energy"].direction == "below"


def test_quality_limits_confidence_but_does_not_change_level(tmp_path):
    service = LongitudinalEmotionService(tmp_path)
    make_sessions(service, 5)
    current_payload = history_session(10, pitch=360, energy=0.22, valence=0.1, arousal=0.92)
    current_payload["quality_score"] = 0.3
    current = service.append("p1", current_payload)[0]
    prior = service.get("p1")[0][:-1]
    report = VoiceChangeEngine().analyze("p1", current, prior)
    assert report.quality_limited is True
    assert report.level == "unusual"
    assert report.confidence < 0.75
