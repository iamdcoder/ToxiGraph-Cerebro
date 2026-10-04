from __future__ import annotations

from app.conversation_graph.dynamics import ConversationDynamicsAnalyzer
from app.conversation_state.engine import (
    ConversationStateAnalyzer,
    ConversationStateConfig,
    ConversationStateError,
)
from app.speaker_diarization.engine import SpeakerAwareAnalysis, SpeakerInteraction, SpeakerSummary, SpeakerTurnAnalysis


def turn(index, speaker, start, end, emotion, valence, arousal, dominance=0.5, confidence=0.9):
    return SpeakerTurnAnalysis(
        turn_index=index,
        speaker=speaker,
        start_seconds=start,
        end_seconds=end,
        duration_seconds=end - start,
        emotion=emotion,
        confidence=confidence,
        probabilities={emotion: confidence, "neutral": 1 - confidence} if emotion != "neutral" else {"neutral": 1.0},
        valence=valence,
        arousal=arousal,
        dominance=dominance,
    )


def analysis(turns):
    interactions = []
    for previous, current in zip(turns, turns[1:], strict=False):
        if previous.speaker == current.speaker:
            continue
        interactions.append(
            SpeakerInteraction(
                previous.speaker,
                current.speaker,
                current.start_seconds,
                max(0.0, current.start_seconds - previous.end_seconds),
                previous.emotion,
                current.emotion,
                current.arousal - previous.arousal,
                current.valence - previous.valence,
                "activated_shift",
            )
        )
    grouped = {}
    for item in turns:
        grouped.setdefault(item.speaker, []).append(item)
    summaries = []
    total_duration = max(item.end_seconds for item in turns)
    for speaker in sorted(grouped):
        items = grouped[speaker]
        seconds = sum(item.duration_seconds for item in items)
        summaries.append(
            SpeakerSummary(
                speaker=speaker,
                turn_count=len(items),
                speaking_seconds=seconds,
                speaking_share=seconds / total_duration,
                dominant_emotion=items[0].emotion,
                dominant_confidence=items[0].confidence,
                mean_confidence=sum(item.confidence for item in items) / len(items),
                first_emotion=items[0].emotion,
                last_emotion=items[-1].emotion,
                emotion_changed=items[0].emotion != items[-1].emotion,
                mean_valence=sum(item.valence for item in items) / len(items),
                mean_arousal=sum(item.arousal for item in items) / len(items),
                mean_dominance=sum(item.dominance for item in items) / len(items),
                first_valence=items[0].valence,
                last_valence=items[-1].valence,
                first_arousal=items[0].arousal,
                last_arousal=items[-1].arousal,
            )
        )
    return SpeakerAwareAnalysis(
        duration_seconds=total_duration,
        sample_rate=16000,
        speaker_count=len(summaries),
        speakers=tuple(summaries),
        turns=tuple(turns),
        interactions=tuple(interactions),
        total_speaking_seconds=sum(item.duration_seconds for item in turns),
        speech_coverage=0.9,
        overlap_seconds=0.0,
        speaker_switches=len(interactions),
        dominant_speaker=summaries[0].speaker,
    )


def dynamics_for(turns):
    return ConversationDynamicsAnalyzer().analyze(analysis(turns))


def test_state_machine_builds_timeline_and_transitions():
    turns = [
        turn(0, "A", 0.0, 1.0, "neutral", 0.8, 0.2),
        turn(1, "B", 1.1, 2.1, "neutral", 0.78, 0.22),
        turn(2, "A", 2.2, 3.2, "angry", 0.45, 0.48),
        turn(3, "B", 3.3, 4.3, "angry", 0.18, 0.76),
        turn(4, "A", 4.4, 5.4, "neutral", 0.55, 0.58),
        turn(5, "B", 5.5, 6.5, "neutral", 0.68, 0.38),
    ]
    result = ConversationStateAnalyzer().analyze(dynamics_for(turns), duration_seconds=6.5)
    assert result.state_count == 5
    assert result.transition_count >= 1
    assert result.start_state in {"calm", "stable", "engaged"}
    assert result.peak_tension > 0
    assert result.peak_tension_at_seconds is not None
    assert abs(sum(result.state_distribution.values()) - 1.0) < 1e-9
    assert abs(sum(result.duration_by_state.values()) - (6.5 - 1.1)) < 1e-9


def test_state_machine_detects_peak_then_cooling():
    turns = [
        turn(0, "A", 0, 1, "neutral", 0.8, 0.2),
        turn(1, "B", 1.05, 2.05, "neutral", 0.6, 0.35),
        turn(2, "A", 2.1, 3.1, "angry", 0.45, 0.55),
        turn(3, "B", 3.15, 4.15, "angry", 0.25, 0.75),
        turn(4, "A", 4.2, 5.2, "angry", 0.1, 0.95),
        turn(5, "B", 5.25, 6.25, "angry", 0.0, 1.0),
        turn(6, "A", 6.3, 7.3, "neutral", 0.75, 0.35),
    ]
    result = ConversationStateAnalyzer().analyze(dynamics_for(turns), duration_seconds=7.3)
    assert any(item.state == "peak_tension" for item in result.points)
    assert any(item.state == "cooling" for item in result.points)
    assert any(item.to_state == "cooling" for item in result.transitions)


def test_state_machine_is_hysteresis_aware():
    turns = [
        turn(0, "A", 0, 1, "neutral", 0.60, 0.40),
        turn(1, "B", 1.1, 2.1, "neutral", 0.56, 0.44),
        turn(2, "A", 2.2, 3.2, "neutral", 0.55, 0.45),
        turn(3, "B", 3.3, 4.3, "neutral", 0.57, 0.43),
    ]
    result = ConversationStateAnalyzer().analyze(dynamics_for(turns))
    states = [item.state for item in result.points]
    assert states.count("escalating") <= 1


def test_state_machine_returns_insufficient_for_no_cross_speaker_events():
    turns = [
        turn(0, "A", 0, 1, "neutral", 0.5, 0.5),
        turn(1, "A", 1.1, 2.1, "happy", 0.7, 0.4),
    ]
    result = ConversationStateAnalyzer().analyze(dynamics_for(turns))
    assert result.start_state == "insufficient_data"
    assert result.state_count == 0
    assert result.transitions == tuple()


def test_state_machine_rejects_invalid_configuration():
    try:
        ConversationStateAnalyzer(ConversationStateConfig(smoothing=0.0))
    except ConversationStateError as exc:
        assert "smoothing" in str(exc).lower()
    else:
        raise AssertionError("Expected invalid smoothing to be rejected")
