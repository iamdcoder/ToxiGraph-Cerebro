from __future__ import annotations

import json

import numpy as np
from fastapi.testclient import TestClient

from app.config import settings
from app.fusion.constants import CANONICAL_LABELS
from app.live.engine import LiveAnalysisError, LiveConfig, LiveInferenceEngine, LiveStreamSession
from app.main import app
from app.routers import live as live_router


client = TestClient(app)


def make_pcm16(duration_seconds: float = 3.1, amplitude: float = 0.12, sample_rate: int = 16000) -> bytes:
    count = int(duration_seconds * sample_rate)
    t = np.arange(count, dtype=np.float32) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * 220.0 * t)
    return (np.clip(signal, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def probabilities(emotion: str) -> dict[str, float]:
    remainder = 0.1 / (len(CANONICAL_LABELS) - 1)
    result = {label: remainder for label in CANONICAL_LABELS}
    result[emotion] = 0.9
    return result


class FakeRuntime:
    def analyze_window(self, session, samples, start_seconds, end_seconds, *, latency_ms):
        from app.live.engine import LiveInferenceResult

        emotion = "angry" if float(np.mean(np.abs(samples))) > 0.18 else "neutral"
        state = "tense" if emotion == "angry" else "calm"
        session.state_machine.current_state = state
        return LiveInferenceResult(
            sequence=session.next_sequence(),
            window_start_seconds=start_seconds,
            window_end_seconds=end_seconds,
            emotion=emotion,
            confidence=0.9,
            probabilities=probabilities(emotion),
            valence=0.75 if emotion == "neutral" else 0.25,
            arousal=0.30 if emotion == "neutral" else 0.88,
            dominance=0.50 if emotion == "neutral" else 0.72,
            state=state,
            state_transition=None,
            speech_ratio=1.0,
            quality_score=0.95,
            quality_verdict="good",
            latency_ms=1.0,
            model_metadata={"fixture": True},
        )


def test_live_config_rejects_invalid_values():
    try:
        LiveStreamSession(16000, LiveConfig(hop_seconds=4.0, window_seconds=3.0))
    except LiveAnalysisError:
        pass
    else:
        raise AssertionError("Expected invalid live configuration to fail")


def test_live_session_decodes_pcm16_and_schedules_rolling_windows():
    session = LiveStreamSession(16000)
    duration = session.append_pcm16(make_pcm16(3.2))
    assert duration == 3.2
    pending = session.pending_window()
    assert pending is not None
    window, start, end = pending
    assert len(window) == 48_000
    assert abs(start - 0.2) < 1e-9
    assert end == 3.2
    assert session.pending_window() is None

    session.append_pcm16(make_pcm16(1.0))
    pending = session.pending_window()
    assert pending is not None
    _, start, end = pending
    assert abs(start - 1.2) < 1e-9
    assert abs(end - 4.2) < 1e-9


def test_live_session_resamples_source_audio():
    session = LiveStreamSession(48000)
    duration = session.append_pcm16(make_pcm16(0.5, sample_rate=48000))
    assert abs(duration - 0.5) < 1 / 16000
    assert len(session.samples) == 8000


def test_live_engine_quality_gates_quiet_audio():
    runtime = LiveInferenceEngine(
        predictor=lambda samples, rate: {
            "emotion": "neutral",
            "confidence": 0.9,
            "probabilities": probabilities("neutral"),
        }
    )
    session = LiveStreamSession(16000)
    samples = np.zeros(48_000, dtype=np.float32)
    result = runtime.analyze_window(session, samples, 0.0, 3.0, latency_ms=0.0)
    assert result.emotion is None
    assert result.state == "quiet"
    assert result.model_metadata["mode"] == "quality_gate"


def test_live_state_machine_reports_cooling_transition():
    from app.live.engine import LiveStateMachine

    machine = LiveStateMachine()
    state, transition = machine.update(
        emotion="angry", valence=0.2, arousal=0.9, speech_active=True
    )
    assert state == "high_tension"
    assert transition is None

    state, transition = machine.update(
        emotion="neutral", valence=0.7, arousal=0.55, speech_active=True
    )
    assert state == "cooling"
    assert transition == "high_tension_to_cooling"


def test_live_status_exposes_stream_contract():
    response = client.get("/api/v1/audio/live/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["mode"] == "single_speaker_realtime"
    assert payload["transport"] == "websocket"
    assert payload["window_seconds"] == 3.0
    assert payload["hop_seconds"] == 1.0


def test_live_websocket_accepts_audio_and_emits_analysis(monkeypatch):
    monkeypatch.setattr(live_router, "get_live_runtime", lambda: FakeRuntime())
    monkeypatch.setattr(live_router.settings, "LIVE_STREAM_ENABLED", True)

    with client.websocket_connect("/api/v1/audio/live/ws") as websocket:
        websocket.send_text(json.dumps({"type": "start", "sample_rate": 16000}))
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["target_sample_rate"] == 16000

        websocket.send_bytes(make_pcm16(3.1))
        analysis = websocket.receive_json()
        assert analysis["type"] == "analysis"
        assert analysis["sequence"] == 1
        assert analysis["emotion"] in {"neutral", "angry"}
        assert 0.0 <= analysis["confidence"] <= 1.0

        websocket.send_text(json.dumps({"type": "stop"}))
        complete = websocket.receive_json()
        assert complete["type"] == "complete"
        assert complete["updates"] == 1


def test_live_websocket_rejects_audio_before_start():
    with client.websocket_connect("/api/v1/audio/live/ws") as websocket:
        websocket.send_bytes(make_pcm16(0.3))
        payload = websocket.receive_json()
        assert payload["type"] == "error"
        assert payload["code"] == "not_started"


def test_live_websocket_unknown_control_is_explicit():
    with client.websocket_connect("/api/v1/audio/live/ws") as websocket:
        websocket.send_text(json.dumps({"type": "bogus"}))
        payload = websocket.receive_json()
        assert payload["type"] == "error"
        assert payload["code"] == "unknown_control"


def test_live_session_rejects_oversized_chunk():
    session = LiveStreamSession(16000, LiveConfig(max_chunk_bytes=32))
    try:
        session.append_pcm16(b"\x00" * 34)
    except LiveAnalysisError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("Expected oversized live chunk to fail")


def test_live_websocket_reports_model_start_failure(monkeypatch):
    monkeypatch.setattr(live_router, "get_live_runtime", lambda: (_ for _ in ()).throw(LiveAnalysisError("Model artifacts are unavailable.")))
    monkeypatch.setattr(live_router.settings, "LIVE_STREAM_ENABLED", True)

    with client.websocket_connect("/api/v1/audio/live/ws") as websocket:
        websocket.send_text(json.dumps({"type": "start", "sample_rate": 16000}))
        payload = websocket.receive_json()
        assert payload["type"] == "error"
        assert payload["code"] == "start_failed"
