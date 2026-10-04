from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient

from app.config import settings
from app.live_multi_speaker.engine import LiveMultiSpeakerConfig, LiveMultiSpeakerError, LiveMultiSpeakerSession
from app.main import app
from app.routers import live_multi_speaker as live_multi_router
from app.speaker_diarization.model import SpeakerTurn

client = TestClient(app)


class SpeakerTurnAnalysisFixture:
    @staticmethod
    def as_analysis(speaker, start_seconds, end_seconds):
        from app.speaker_diarization.engine import SpeakerTurnAnalysis
        return SpeakerTurnAnalysis(
            turn_index=0,
            speaker=speaker,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            duration_seconds=end_seconds - start_seconds,
            emotion="neutral",
            confidence=0.8,
            probabilities={"neutral": 0.8, "angry": 0.2},
            valence=0.5,
            arousal=0.5,
            dominance=0.5,
        )


def make_pcm16(duration_seconds: float = 3.2, amplitude: float = 0.10, sample_rate: int = 16000) -> bytes:
    count = int(duration_seconds * sample_rate)
    t = np.arange(count, dtype=np.float32) / sample_rate
    signal = amplitude * np.sin(2 * np.pi * 220.0 * t)
    return (np.clip(signal, -1.0, 1.0) * 32767.0).astype('<i2').tobytes()


def fake_prediction(samples, sample_rate):
    return {
        "emotion": "angry" if len(samples) > 18_000 else "neutral",
        "confidence": 0.85,
        "probabilities": {"angry": 0.85, "neutral": 0.15},
        "valence": 0.25,
        "arousal": 0.80,
        "dominance": 0.60,
    }


def test_has_analysis_compares_against_current_turn_boundary():
    session = LiveMultiSpeakerSession(16_000, LiveMultiSpeakerConfig(context_seconds=4.0))
    session.analyses = [
        SpeakerTurnAnalysisFixture.as_analysis("SPEAKER_01", 0.0, 1.0),
    ]
    candidate = SpeakerTurn("SPEAKER_01", 0.0, 1.5, 1.5)
    assert session._has_analysis(candidate) is True
    different = SpeakerTurn("SPEAKER_01", 1.0, 3.0, 2.0)
    assert session._has_analysis(different) is False


def test_multi_config_rejects_invalid_values():
    with np.testing.assert_raises(LiveMultiSpeakerError):
        LiveMultiSpeakerSession(16_000, LiveMultiSpeakerConfig(context_seconds=1.0))
    with np.testing.assert_raises(LiveMultiSpeakerError):
        LiveMultiSpeakerSession(16_000, LiveMultiSpeakerConfig(stabilization_seconds=8.0))


def test_multi_session_decodes_and_schedules_context():
    session = LiveMultiSpeakerSession(
        16_000,
        LiveMultiSpeakerConfig(context_seconds=4.0, hop_seconds=1.0, min_context_seconds=2.0),
    )
    session.append_pcm16(make_pcm16(4.1))
    pending = session.pending_window()
    assert pending is not None
    window, start, end = pending
    assert len(window) == 64_000
    assert abs(start - 0.1) < 1e-9
    assert abs(end - 4.1) < 1e-9
    assert session.pending_window() is None


def test_multi_session_resamples_source_audio():
    session = LiveMultiSpeakerSession(
        48_000,
        LiveMultiSpeakerConfig(context_seconds=4.0, min_context_seconds=2.0),
    )
    session.append_pcm16(make_pcm16(1.0, sample_rate=48_000))
    assert len(session.samples) == 16_000


def test_multi_session_rejects_odd_pcm_chunk():
    session = LiveMultiSpeakerSession(16_000)
    with np.testing.assert_raises(LiveMultiSpeakerError):
        session.append_pcm16(b"\x00")


def test_speaker_ids_are_stabilized_by_overlap():
    session = LiveMultiSpeakerSession(16_000, LiveMultiSpeakerConfig(context_seconds=4.0))
    session.global_turns = [
        SpeakerTurn("SPEAKER_01", 0.0, 1.8, 1.8),
        SpeakerTurn("SPEAKER_02", 2.0, 3.6, 1.6),
    ]
    resolved = session._stabilize_speaker_ids([
        SpeakerTurn("LOCAL_A", 0.1, 1.7, 1.6),
        SpeakerTurn("LOCAL_B", 2.1, 3.4, 1.3),
    ])
    assert [item.speaker for item in resolved] == ["SPEAKER_01", "SPEAKER_02"]
    assert session.speaker_match_hits == 2


def test_new_local_speaker_gets_new_stable_identity():
    session = LiveMultiSpeakerSession(16_000, LiveMultiSpeakerConfig(context_seconds=4.0))
    resolved = session._stabilize_speaker_ids([SpeakerTurn("LOCAL_A", 0.0, 1.2, 1.2)])
    assert resolved[0].speaker == "SPEAKER_01"


def test_merge_into_global_turns_extends_existing_boundary():
    session = LiveMultiSpeakerSession(16_000, LiveMultiSpeakerConfig(context_seconds=4.0))
    session.global_turns = [SpeakerTurn("SPEAKER_01", 0.0, 1.5, 1.5)]
    session._merge_into_global_turns([SpeakerTurn("SPEAKER_01", 1.2, 1.9, 0.7)])
    assert len(session.global_turns) == 1
    assert session.global_turns[0].end_seconds == 1.9


def test_analysis_marks_recent_turns_as_provisional():
    session = LiveMultiSpeakerSession(
        16_000,
        LiveMultiSpeakerConfig(context_seconds=4.0, stabilization_seconds=0.8, min_turn_seconds=0.5),
    )
    turns = [
        SpeakerTurn("A", 0.0, 1.6, 1.6),
        SpeakerTurn("B", 1.8, 2.8, 1.0),
        SpeakerTurn("A", 3.0, 4.0, 1.0),
    ]
    session.append_pcm16((np.ones(64_000, dtype=np.float32) * 0.05 * 32767).astype("<i2").tobytes())
    result = session.analyze_window(
        np.ones(64_000, dtype=np.float32) * 0.05,
        0.0,
        4.0,
        diarizer=lambda samples, rate: turns,
        predictor=fake_prediction,
    )
    provisional = [item for item in result.turns if item["status"] == "provisional"]
    assert provisional
    assert result.provisional is True


def test_analysis_produces_conversation_graph_and_state():
    session = LiveMultiSpeakerSession(
        16_000,
        LiveMultiSpeakerConfig(context_seconds=4.0, stabilization_seconds=0.4, min_turn_seconds=0.5),
    )
    turns = [
        SpeakerTurn("A", 0.0, 1.3, 1.3),
        SpeakerTurn("B", 1.5, 2.8, 1.3),
        SpeakerTurn("A", 3.0, 3.4, 0.4),
    ]
    session.append_pcm16((np.ones(64_000, dtype=np.float32) * 0.05 * 32767).astype("<i2").tobytes())
    result = session.analyze_window(
        np.ones(64_000, dtype=np.float32) * 0.05,
        0.0,
        4.0,
        diarizer=lambda samples, rate: turns,
        predictor=fake_prediction,
    )
    assert result.speaker_count == 2
    assert result.interaction_graph["metrics"]["edge_count"] >= 1
    assert result.conversation_dynamics["event_count"] >= 1
    assert result.conversation_state["points"]


def test_analysis_keeps_persistent_ids_when_local_labels_swap():
    session = LiveMultiSpeakerSession(
        16_000,
        LiveMultiSpeakerConfig(context_seconds=4.0, stabilization_seconds=0.4, min_turn_seconds=0.5),
    )
    first = [SpeakerTurn("LOCAL_A", 0.0, 1.6, 1.6), SpeakerTurn("LOCAL_B", 1.8, 3.2, 1.4)]
    session.append_pcm16((np.ones(64_000, dtype=np.float32) * 0.05 * 32767).astype("<i2").tobytes())
    session.analyze_window(np.ones(64_000, dtype=np.float32) * 0.05, 0.0, 4.0, diarizer=lambda *_: first, predictor=fake_prediction)
    second = [SpeakerTurn("LOCAL_B", 0.0, 1.5, 1.5), SpeakerTurn("LOCAL_A", 1.7, 3.5, 1.8)]
    session.analyze_window(np.ones(64_000, dtype=np.float32) * 0.05, 1.0, 5.0, diarizer=lambda *_: second, predictor=fake_prediction)
    speakers = {turn.speaker for turn in session.global_turns}
    assert speakers == {"SPEAKER_01", "SPEAKER_02"}
    assert session.speaker_match_hits >= 2


def test_multi_status_exposes_rolling_contract(monkeypatch):
    monkeypatch.setattr(settings, "LIVE_MULTI_SPEAKER_ENABLED", True)
    monkeypatch.setattr(settings, "SPEAKER_DIARIZATION_ENABLED", True)
    response = client.get("/api/v1/audio/live-multi/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["mode"] == "rolling_multi_speaker_realtime"
    assert payload["transport"] == "websocket"
    assert payload["context_seconds"] == 8.0


def test_multi_websocket_emits_speaker_aware_analysis(monkeypatch):
    class FakeDiarizer:
        model_id = "fixture-diarizer"

        def diarize(self, samples, sample_rate):
            return [
                SpeakerTurn("A", 0.0, 1.3, 1.3),
                SpeakerTurn("B", 1.5, 2.8, 1.3),
                SpeakerTurn("A", 3.0, 3.2, 0.2),
            ]

    runtime = {"diarizer": FakeDiarizer(), "predictor": fake_prediction, "metadata": {"fixture": True}}
    monkeypatch.setattr(live_multi_router, "get_multi_runtime", lambda: runtime)
    monkeypatch.setattr(live_multi_router.settings, "LIVE_MULTI_SPEAKER_ENABLED", True)
    monkeypatch.setattr(live_multi_router.settings, "SPEAKER_DIARIZATION_ENABLED", True)
    monkeypatch.setattr(live_multi_router.settings, "LIVE_MULTI_SPEAKER_CONTEXT_SECONDS", 3.0)
    monkeypatch.setattr(live_multi_router.settings, "LIVE_MULTI_SPEAKER_MIN_CONTEXT_SECONDS", 2.0)
    monkeypatch.setattr(live_multi_router.settings, "LIVE_MULTI_SPEAKER_STABILIZATION_SECONDS", 0.2)
    monkeypatch.setattr(live_multi_router.settings, "LIVE_MULTI_SPEAKER_MAX_DURATION_SECONDS", 20.0)

    with client.websocket_connect("/api/v1/audio/live-multi/ws") as websocket:
        websocket.send_text(json.dumps({"type": "start", "sample_rate": 16_000}))
        ready = websocket.receive_json()
        assert ready["type"] == "ready"
        assert ready["mode"] == "rolling_multi_speaker_realtime"

        websocket.send_bytes(make_pcm16(3.1))
        analysis = websocket.receive_json()
        assert analysis["type"] == "analysis"
        assert analysis["speaker_count"] == 2
        assert set(analysis["speakers"][0].keys()) >= {"speaker", "turn_count", "speaking_share"}
        assert analysis["interaction_graph"]["metrics"]["edge_count"] >= 1

        websocket.send_text(json.dumps({"type": "stop"}))
        complete = websocket.receive_json()
        assert complete["type"] == "complete"
        assert complete["updates"] == 1


def test_multi_websocket_rejects_audio_before_start():
    with client.websocket_connect("/api/v1/audio/live-multi/ws") as websocket:
        websocket.send_bytes(make_pcm16(0.2))
        payload = websocket.receive_json()
        assert payload["type"] == "error"
        assert payload["code"] == "not_started"
