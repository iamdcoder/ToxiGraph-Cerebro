from types import SimpleNamespace

from app.conversation_graph.engine import ConversationGraphError, build_conversation_graph
from app.speaker_diarization.engine import SpeakerAwareAnalysis, SpeakerInteraction, SpeakerSummary, SpeakerTurnAnalysis


def turn(index, speaker, start, end, emotion, confidence, valence, arousal):
    return SpeakerTurnAnalysis(
        turn_index=index, speaker=speaker, start_seconds=start, end_seconds=end,
        duration_seconds=end-start, emotion=emotion, confidence=confidence,
        probabilities={emotion: confidence, "neutral": 1-confidence} if emotion != "neutral" else {"neutral": 1.0},
        valence=valence, arousal=arousal, dominance=0.5,
    )


def summary(speaker, turns):
    return SpeakerSummary(
        speaker=speaker, turn_count=len(turns), speaking_seconds=sum(t.duration_seconds for t in turns),
        speaking_share=sum(t.duration_seconds for t in turns) / 4.0, dominant_emotion=turns[0].emotion,
        dominant_confidence=turns[0].confidence, mean_confidence=sum(t.confidence for t in turns) / len(turns),
        first_emotion=turns[0].emotion, last_emotion=turns[-1].emotion, emotion_changed=turns[0].emotion != turns[-1].emotion,
        mean_valence=sum(t.valence for t in turns if t.valence is not None) / len(turns),
        mean_arousal=sum(t.arousal for t in turns if t.arousal is not None) / len(turns),
        mean_dominance=0.5, first_valence=turns[0].valence, last_valence=turns[-1].valence,
        first_arousal=turns[0].arousal, last_arousal=turns[-1].arousal,
    )


def make_analysis(turns, interactions):
    grouped = {}
    for item in turns:
        grouped.setdefault(item.speaker, []).append(item)
    summaries = tuple(summary(name, items) for name, items in grouped.items())
    return SpeakerAwareAnalysis(
        duration_seconds=4.0, sample_rate=16000, speaker_count=len(summaries), speakers=summaries,
        turns=tuple(turns), interactions=tuple(interactions), total_speaking_seconds=3.8, speech_coverage=.95,
        overlap_seconds=0.0, speaker_switches=len(interactions), dominant_speaker="A",
    )


def test_graph_builds_nodes_and_directional_edges():
    turns = [
        turn(0, "A", 0.0, 1.0, "neutral", .9, .7, .3),
        turn(1, "B", 1.1, 2.1, "neutral", .8, .6, .4),
        turn(2, "A", 2.2, 3.2, "angry", .85, .2, .8),
        turn(3, "B", 3.25, 4.0, "angry", .9, .15, .9),
    ]
    interactions = [
        SpeakerInteraction("A", "B", 1.1, .1, "neutral", "neutral", .1, -.1, "continuity"),
        SpeakerInteraction("B", "A", 2.2, .1, "neutral", "angry", .4, -.4, "activated_shift"),
        SpeakerInteraction("A", "B", 3.25, .05, "angry", "angry", .1, -.05, "continuity"),
    ]
    graph = build_conversation_graph(make_analysis(turns, interactions))
    assert graph.node_count == 2
    assert graph.edge_count == 2
    assert graph.total_interactions == 3
    assert graph.reciprocity == 1.0
    edge_ab = next(e for e in graph.edges if e.from_speaker == "A" and e.to_speaker == "B")
    assert edge_ab.interaction_count == 2
    assert edge_ab.interaction_weight == 1.0
    assert edge_ab.emotion_transition_count == 0
    assert edge_ab.continuity_count == 2


def test_graph_aggregates_affect_changes_and_latency():
    turns = [turn(0,"A",0,1,"neutral",.9,.8,.2), turn(1,"B",1.5,2.5,"angry",.9,.2,.8)]
    interactions = [SpeakerInteraction("A","B",1.5,.5,"neutral","angry",.6,-.6,"activated_shift")]
    graph = build_conversation_graph(make_analysis(turns, interactions))
    edge = graph.edges[0]
    assert edge.mean_gap_seconds == .5
    assert edge.mean_arousal_change == .6
    assert edge.mean_valence_change == -.6
    assert edge.activated_shift_count == 1


def test_graph_density_for_three_speakers():
    turns = [
        turn(0,"A",0,1,"neutral",.9,.5,.4), turn(1,"B",1,2,"neutral",.9,.5,.4),
        turn(2,"C",2,3,"neutral",.9,.5,.4),
    ]
    interactions = [
        SpeakerInteraction("A","B",1,0,"neutral","neutral",0,0,"continuity"),
        SpeakerInteraction("B","C",2,0,"neutral","neutral",0,0,"continuity"),
    ]
    graph = build_conversation_graph(make_analysis(turns, interactions))
    assert graph.edge_count == 2
    assert graph.density == 2/6
    assert graph.connected_speakers == 3


def test_graph_rejects_empty_turns():
    analysis = SpeakerAwareAnalysis(
        duration_seconds=1.0, sample_rate=16000, speaker_count=0, speakers=tuple(), turns=tuple(),
        interactions=tuple(), total_speaking_seconds=0.0, speech_coverage=0.0, overlap_seconds=0.0,
        speaker_switches=0, dominant_speaker=None,
    )
    try:
        build_conversation_graph(analysis)
    except ConversationGraphError as exc:
        assert "without speaker turns" in str(exc)
    else:
        raise AssertionError("Expected graph construction to fail")


def test_graph_semantics_are_non_causal_and_json_ready():
    turns = [turn(0,"A",0,1,"neutral",.9,.5,.4), turn(1,"B",1,2,"angry",.9,.2,.7)]
    interactions = [SpeakerInteraction("A","B",1,0,"neutral","angry",.3,-.3,"activated_shift")]
    payload = build_conversation_graph(make_analysis(turns, interactions)).to_dict()
    assert payload["graph_type"] == "directed_conversation_interaction"
    assert payload["semantics"]["affect_change"].endswith("not a causal attribution")
    assert len(payload["nodes"]) == 2
    assert len(payload["edges"]) == 1
