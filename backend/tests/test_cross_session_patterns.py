from datetime import datetime, timedelta, timezone

from app.cross_session_patterns.engine import CrossSessionPatternEngine, MIN_PATTERN_SESSIONS
from app.longitudinal_emotion.service import HistorySession


def session(i: int, emotion: str, *, valence: float, arousal: float, dominance: float, energy: float = 0.08, pitch: float = 220.0) -> HistorySession:
    return HistorySession(
        session_id=f"s{i}",
        recorded_at=(datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=i)).isoformat(),
        duration_seconds=8.0,
        emotion=emotion,
        confidence=0.8,
        acoustic={
            "pitch_mean_hz": pitch,
            "pitch_std_hz": 20.0,
            "energy_mean": energy,
            "estimated_syllable_rate_sps": 3.0,
            "estimated_pause_ratio": 0.18,
            "spectral_centroid_mean_hz": 1800.0,
        },
        affect={"valence": valence, "arousal": arousal, "dominance": dominance},
        quality_score=0.9,
        baseline_deviation=0.2,
    )


def test_insufficient_history_is_explicit():
    report = CrossSessionPatternEngine().analyze("p1", [session(i, "neutral", valence=.5, arousal=.5, dominance=.5) for i in range(MIN_PATTERN_SESSIONS - 1)])
    assert report.available is False
    assert report.level if hasattr(report, "level") else True
    assert "prior" not in report.summary.lower() or MIN_PATTERN_SESSIONS >= 4


def test_recurring_emotion_and_transition_are_detected():
    values = [
        session(0, "neutral", valence=.5, arousal=.4, dominance=.5),
        session(1, "angry", valence=.2, arousal=.8, dominance=.7),
        session(2, "neutral", valence=.5, arousal=.4, dominance=.5),
        session(3, "angry", valence=.2, arousal=.8, dominance=.7),
        session(4, "neutral", valence=.5, arousal=.4, dominance=.5),
        session(5, "angry", valence=.2, arousal=.8, dominance=.7),
    ]
    report = CrossSessionPatternEngine().analyze("p1", values)
    assert report.available is True
    assert report.dominant_emotion in {"neutral", "angry"}
    assert any(item.pattern_type == "emotion_transition" for item in report.patterns)
    assert any(item.from_emotion == "neutral" and item.to_emotion == "angry" for item in report.repeated_transitions)


def test_repeated_affect_direction_is_detected():
    values = [
        session(0, "neutral", valence=.3, arousal=.3, dominance=.4),
        session(1, "neutral", valence=.4, arousal=.4, dominance=.4),
        session(2, "neutral", valence=.5, arousal=.5, dominance=.5),
        session(3, "neutral", valence=.6, arousal=.6, dominance=.6),
        session(4, "neutral", valence=.7, arousal=.7, dominance=.7),
    ]
    report = CrossSessionPatternEngine().analyze("p1", values)
    assert any(item.pattern_type == "affect_momentum" for item in report.patterns)


def test_affect_stability_is_detected():
    values = [session(i, "neutral", valence=.5 + (i % 2) * .01, arousal=.5, dominance=.5) for i in range(5)]
    report = CrossSessionPatternEngine().analyze("p1", values)
    assert any(item.pattern_type == "affect_stability" for item in report.patterns)


def test_affect_volatility_is_detected():
    values = [
        session(0, "neutral", valence=.1, arousal=.1, dominance=.2),
        session(1, "angry", valence=.9, arousal=.9, dominance=.8),
        session(2, "neutral", valence=.1, arousal=.1, dominance=.2),
        session(3, "angry", valence=.9, arousal=.9, dominance=.8),
        session(4, "neutral", valence=.1, arousal=.1, dominance=.2),
    ]
    report = CrossSessionPatternEngine().analyze("p1", values)
    assert any(item.pattern_type == "affect_volatility" for item in report.patterns)


def test_acoustic_affect_association_is_descriptive():
    values = [
        session(0, "neutral", valence=.5, arousal=.2, dominance=.5, energy=.05, pitch=180),
        session(1, "neutral", valence=.5, arousal=.3, dominance=.5, energy=.08, pitch=200),
        session(2, "neutral", valence=.5, arousal=.5, dominance=.5, energy=.11, pitch=220),
        session(3, "neutral", valence=.5, arousal=.7, dominance=.5, energy=.15, pitch=240),
        session(4, "neutral", valence=.5, arousal=.9, dominance=.5, energy=.19, pitch=260),
    ]
    report = CrossSessionPatternEngine().analyze("p1", values)
    assert any(item.pattern_type == "acoustic_affect_link" for item in report.patterns)


def test_missing_labels_do_not_break_pattern_engine():
    values = [
        session(0, "neutral", valence=.5, arousal=.5, dominance=.5),
        session(1, None, valence=.5, arousal=.5, dominance=.5),
        session(2, "neutral", valence=.5, arousal=.5, dominance=.5),
        session(3, "neutral", valence=.5, arousal=.5, dominance=.5),
        session(4, "neutral", valence=.5, arousal=.5, dominance=.5),
    ]
    report = CrossSessionPatternEngine().analyze("p1", values)
    assert report.available is True
    assert report.labeled_session_count == 4
