from __future__ import annotations

from app.conversation_graph.dynamics import ConversationDynamicsAnalyzer, ConversationDynamicsConfig, ConversationDynamicsError
from app.speaker_diarization.engine import SpeakerAwareAnalysis, SpeakerInteraction, SpeakerSummary, SpeakerTurnAnalysis


def make_turn(index, speaker, start, end, emotion, valence, arousal, dominance=0.5, confidence=0.9):
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


def make_analysis(turns):
    interactions = []
    for previous, current in zip(turns, turns[1:], strict=False):
        if previous.speaker == current.speaker:
            continue
        interactions.append(
            SpeakerInteraction(
                from_speaker=previous.speaker,
                to_speaker=current.speaker,
                at_seconds=current.start_seconds,
                gap_seconds=max(0.0, current.start_seconds - previous.end_seconds),
                previous_emotion=previous.emotion,
                next_emotion=current.emotion,
                arousal_change=(current.arousal - previous.arousal),
                valence_change=(current.valence - previous.valence),
                type="activated_shift",
            )
        )
    grouped = {}
    for turn in turns:
        grouped.setdefault(turn.speaker, []).append(turn)
    speakers = []
    total_duration = max(turn.end_seconds for turn in turns)
    for speaker in sorted(grouped):
        items = grouped[speaker]
        seconds = sum(item.duration_seconds for item in items)
        speakers.append(
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
        speaker_count=len(speakers),
        speakers=tuple(speakers),
        turns=tuple(turns),
        interactions=tuple(interactions),
        total_speaking_seconds=sum(turn.duration_seconds for turn in turns),
        speech_coverage=0.9,
        overlap_seconds=0.0,
        speaker_switches=len(interactions),
        dominant_speaker=speakers[0].speaker if speakers else None,
    )


def test_dynamics_detects_escalation_and_deescalation():
    turns = [
        make_turn(0, "A", 0.0, 1.0, "neutral", 0.8, 0.2),
        make_turn(1, "B", 1.1, 2.1, "angry", 0.5, 0.5),
        make_turn(2, "A", 2.3, 3.3, "angry", 0.3, 0.8),
        make_turn(3, "B", 3.9, 4.9, "neutral", 0.7, 0.35),
    ]
    result = ConversationDynamicsAnalyzer().analyze(make_analysis(turns))
    assert result.event_count == 3
    assert result.escalation_events == 2
    assert result.deescalation_events == 1
    assert result.dominant_pattern == "escalating"
    assert result.net_tension > 0
    assert result.arousal_rise_events == 2
    assert result.valence_drop_events == 2


def test_dynamics_tracks_latency_and_response_speed():
    turns = [
        make_turn(0, "A", 0.0, 1.0, "neutral", 0.5, 0.4),
        make_turn(1, "B", 1.2, 2.2, "neutral", 0.5, 0.4),
        make_turn(2, "A", 2.7, 3.7, "neutral", 0.5, 0.4),
        make_turn(3, "B", 5.8, 6.8, "neutral", 0.5, 0.4),
    ]
    result = ConversationDynamicsAnalyzer().analyze(make_analysis(turns))
    assert abs(result.mean_response_gap_seconds - 0.9333333333) < 1e-9
    assert result.median_response_gap_seconds == 0.5
    assert result.p90_response_gap_seconds > 1.0
    assert result.fast_response_share < 1.0
    assert result.response_latency_shift_seconds > 0


def test_dynamics_measures_affect_synchrony_as_state_alignment():
    turns = [
        make_turn(0, "A", 0.0, 1.0, "neutral", 0.6, 0.5, 0.4),
        make_turn(1, "B", 1.1, 2.1, "neutral", 0.6, 0.5, 0.4),
        make_turn(2, "A", 2.2, 3.2, "neutral", 0.61, 0.51, 0.41),
    ]
    result = ConversationDynamicsAnalyzer().analyze(make_analysis(turns))
    assert result.affect_synchrony is not None
    assert result.affect_synchrony > 0.95
    assert result.events[0].affect_alignment > 0.99


def test_dynamics_detects_high_volatility_and_low_stability():
    turns = [
        make_turn(0, "A", 0.0, 1.0, "neutral", 0.1, 0.1),
        make_turn(1, "B", 1.1, 2.1, "angry", 0.9, 0.9),
        make_turn(2, "A", 2.2, 3.2, "sad", 0.1, 0.1),
    ]
    result = ConversationDynamicsAnalyzer().analyze(make_analysis(turns))
    assert result.volatility > 0.4
    assert result.stability_score < 0.6


def test_dynamics_returns_insufficient_data_for_single_speaker():
    turns = [
        make_turn(0, "A", 0.0, 1.0, "neutral", 0.5, 0.5),
        make_turn(1, "A", 1.1, 2.1, "happy", 0.7, 0.5),
    ]
    result = ConversationDynamicsAnalyzer().analyze(make_analysis(turns))
    assert result.event_count == 0
    assert result.dominant_pattern == "insufficient_data"
    assert result.affect_synchrony is None


def test_dynamics_rejects_empty_analysis():
    analysis = SpeakerAwareAnalysis(
        duration_seconds=1.0,
        sample_rate=16000,
        speaker_count=0,
        speakers=tuple(),
        turns=tuple(),
        interactions=tuple(),
        total_speaking_seconds=0.0,
        speech_coverage=0.0,
        overlap_seconds=0.0,
        speaker_switches=0,
        dominant_speaker=None,
    )
    try:
        ConversationDynamicsAnalyzer().analyze(analysis)
    except ConversationDynamicsError as exc:
        assert "without speaker turns" in str(exc)
    else:
        raise AssertionError("Expected conversation dynamics to reject empty turns")


def test_custom_thresholds_change_event_classification():
    turns = [
        make_turn(0, "A", 0.0, 1.0, "neutral", 0.52, 0.45),
        make_turn(1, "B", 1.1, 2.1, "neutral", 0.44, 0.51),
    ]
    result = ConversationDynamicsAnalyzer(
        ConversationDynamicsConfig(
            escalation_arousal_delta=0.04,
            escalation_valence_delta=-0.04,
            deescalation_arousal_delta=-0.04,
            deescalation_valence_delta=0.04,
        )
    ).analyze(make_analysis(turns))
    assert result.escalation_events == 1
    assert result.events[0].event_type == "escalation"
