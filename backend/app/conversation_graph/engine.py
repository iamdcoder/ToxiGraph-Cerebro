from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from app.speaker_diarization.engine import SpeakerAwareAnalysis, SpeakerTurnAnalysis


class ConversationGraphError(ValueError):
    """Raised when an interaction graph cannot be constructed."""


@dataclass(frozen=True)
class GraphNode:
    speaker: str
    turn_count: int
    speaking_seconds: float
    speaking_share: float
    dominant_emotion: str
    mean_confidence: float
    mean_valence: float | None
    mean_arousal: float | None
    mean_dominance: float | None
    in_interactions: int
    out_interactions: int
    unique_contacts: int
    first_turn_seconds: float
    last_turn_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "speaker": self.speaker,
            "turn_count": self.turn_count,
            "speaking_seconds": round(self.speaking_seconds, 4),
            "speaking_share": round(self.speaking_share, 6),
            "dominant_emotion": self.dominant_emotion,
            "mean_confidence": round(self.mean_confidence, 6),
            "mean_valence": _round_optional(self.mean_valence),
            "mean_arousal": _round_optional(self.mean_arousal),
            "mean_dominance": _round_optional(self.mean_dominance),
            "in_interactions": self.in_interactions,
            "out_interactions": self.out_interactions,
            "unique_contacts": self.unique_contacts,
            "first_turn_seconds": round(self.first_turn_seconds, 4),
            "last_turn_seconds": round(self.last_turn_seconds, 4),
        }


@dataclass(frozen=True)
class GraphEdge:
    from_speaker: str
    to_speaker: str
    interaction_count: int
    interaction_weight: float
    mean_gap_seconds: float
    min_gap_seconds: float
    max_gap_seconds: float
    continuity_count: int
    emotion_transition_count: int
    activated_shift_count: int
    mean_valence_change: float | None
    mean_arousal_change: float | None
    mean_abs_valence_change: float | None
    mean_abs_arousal_change: float | None
    first_interaction_seconds: float
    last_interaction_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_speaker": self.from_speaker,
            "to_speaker": self.to_speaker,
            "interaction_count": self.interaction_count,
            "interaction_weight": round(self.interaction_weight, 6),
            "mean_gap_seconds": round(self.mean_gap_seconds, 4),
            "min_gap_seconds": round(self.min_gap_seconds, 4),
            "max_gap_seconds": round(self.max_gap_seconds, 4),
            "continuity_count": self.continuity_count,
            "emotion_transition_count": self.emotion_transition_count,
            "activated_shift_count": self.activated_shift_count,
            "mean_valence_change": _round_optional(self.mean_valence_change),
            "mean_arousal_change": _round_optional(self.mean_arousal_change),
            "mean_abs_valence_change": _round_optional(self.mean_abs_valence_change),
            "mean_abs_arousal_change": _round_optional(self.mean_abs_arousal_change),
            "first_interaction_seconds": round(self.first_interaction_seconds, 4),
            "last_interaction_seconds": round(self.last_interaction_seconds, 4),
        }


@dataclass(frozen=True)
class ConversationInteractionGraph:
    nodes: tuple[GraphNode, ...]
    edges: tuple[GraphEdge, ...]
    node_count: int
    edge_count: int
    total_interactions: int
    density: float
    reciprocity: float
    active_pairs: int
    connected_speakers: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_type": "directed_conversation_interaction",
            "directed": True,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "metrics": {
                "node_count": self.node_count,
                "edge_count": self.edge_count,
                "total_interactions": self.total_interactions,
                "density": round(self.density, 6),
                "reciprocity": round(self.reciprocity, 6),
                "active_pairs": self.active_pairs,
                "connected_speakers": self.connected_speakers,
            },
            "semantics": {
                "edge_direction": "observed turn-to-turn response direction",
                "edge_weight": "share of all cross-speaker responses represented by this edge",
                "affect_change": "change between consecutive speakers' analyzed turns; not a causal attribution",
            },
        }


@dataclass
class _EdgeAccumulator:
    from_speaker: str
    to_speaker: str
    gaps: list[float]
    valence_changes: list[float]
    arousal_changes: list[float]
    types: list[str]
    times: list[float]

    @property
    def interaction_count(self) -> int:
        return len(self.gaps)


def _round_optional(value: float | None) -> float | None:
    return round(float(value), 6) if value is not None else None


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _speaker_turns(analysis: SpeakerAwareAnalysis) -> dict[str, list[SpeakerTurnAnalysis]]:
    grouped: dict[str, list[SpeakerTurnAnalysis]] = {}
    for turn in analysis.turns:
        grouped.setdefault(turn.speaker, []).append(turn)
    return grouped


def build_conversation_graph(analysis: SpeakerAwareAnalysis) -> ConversationInteractionGraph:
    """Build a deterministic directed response graph from analyzed speaker turns."""
    if not analysis.turns:
        raise ConversationGraphError("Cannot build an interaction graph without speaker turns.")

    grouped = _speaker_turns(analysis)
    speaker_names = sorted(grouped)
    if not speaker_names:
        raise ConversationGraphError("No speakers are available for graph construction.")

    interactions_by_pair: dict[tuple[str, str], _EdgeAccumulator] = {}
    total_cross = 0
    for interaction in analysis.interactions:
        pair = (interaction.from_speaker, interaction.to_speaker)
        accumulator = interactions_by_pair.get(pair)
        if accumulator is None:
            accumulator = _EdgeAccumulator(
                from_speaker=interaction.from_speaker,
                to_speaker=interaction.to_speaker,
                gaps=[],
                valence_changes=[],
                arousal_changes=[],
                types=[],
                times=[],
            )
            interactions_by_pair[pair] = accumulator
        accumulator.gaps.append(float(interaction.gap_seconds))
        if interaction.valence_change is not None:
            accumulator.valence_changes.append(float(interaction.valence_change))
        if interaction.arousal_change is not None:
            accumulator.arousal_changes.append(float(interaction.arousal_change))
        accumulator.types.append(interaction.type)
        accumulator.times.append(float(interaction.at_seconds))
        total_cross += 1

    max_count = max((item.interaction_count for item in interactions_by_pair.values()), default=1)
    reciprocal_pairs = sum(
        1
        for source, target in interactions_by_pair
        if (target, source) in interactions_by_pair and source < target
    )
    possible_edges = len(speaker_names) * max(len(speaker_names) - 1, 1)
    density = len(interactions_by_pair) / possible_edges if len(speaker_names) > 1 else 0.0
    reciprocity = (2 * reciprocal_pairs / len(interactions_by_pair)) if interactions_by_pair else 0.0

    nodes: list[GraphNode] = []
    indegree = {speaker: 0 for speaker in speaker_names}
    outdegree = {speaker: 0 for speaker in speaker_names}
    contacts = {speaker: set() for speaker in speaker_names}
    for (source, target), accumulator in interactions_by_pair.items():
        indegree[target] += accumulator.interaction_count
        outdegree[source] += accumulator.interaction_count
        contacts[source].add(target)
        contacts[target].add(source)

    summary_by_speaker = {item.speaker: item for item in analysis.speakers}
    for speaker in speaker_names:
        summary = summary_by_speaker.get(speaker)
        if summary is None:
            raise ConversationGraphError(f"Missing speaker summary for '{speaker}'.")
        turns = sorted(grouped[speaker], key=lambda item: (item.start_seconds, item.turn_index))
        nodes.append(
            GraphNode(
                speaker=speaker,
                turn_count=summary.turn_count,
                speaking_seconds=summary.speaking_seconds,
                speaking_share=summary.speaking_share,
                dominant_emotion=summary.dominant_emotion,
                mean_confidence=summary.mean_confidence,
                mean_valence=summary.mean_valence,
                mean_arousal=summary.mean_arousal,
                mean_dominance=summary.mean_dominance,
                in_interactions=indegree[speaker],
                out_interactions=outdegree[speaker],
                unique_contacts=len(contacts[speaker]),
                first_turn_seconds=turns[0].start_seconds,
                last_turn_seconds=turns[-1].end_seconds,
            )
        )

    edges: list[GraphEdge] = []
    for pair in sorted(interactions_by_pair):
        accumulator = interactions_by_pair[pair]
        edge_types = accumulator.types
        edges.append(
            GraphEdge(
                from_speaker=accumulator.from_speaker,
                to_speaker=accumulator.to_speaker,
                interaction_count=accumulator.interaction_count,
                interaction_weight=accumulator.interaction_count / max(max_count, 1),
                mean_gap_seconds=float(_mean(accumulator.gaps) or 0.0),
                min_gap_seconds=min(accumulator.gaps),
                max_gap_seconds=max(accumulator.gaps),
                continuity_count=sum(item == "continuity" for item in edge_types),
                emotion_transition_count=sum(item == "emotion_transition" for item in edge_types),
                activated_shift_count=sum(item == "activated_shift" for item in edge_types),
                mean_valence_change=_mean(accumulator.valence_changes),
                mean_arousal_change=_mean(accumulator.arousal_changes),
                mean_abs_valence_change=_mean([abs(item) for item in accumulator.valence_changes]),
                mean_abs_arousal_change=_mean([abs(item) for item in accumulator.arousal_changes]),
                first_interaction_seconds=min(accumulator.times),
                last_interaction_seconds=max(accumulator.times),
            )
        )

    connected = sum(1 for speaker in speaker_names if contacts[speaker])
    return ConversationInteractionGraph(
        nodes=tuple(nodes),
        edges=tuple(edges),
        node_count=len(nodes),
        edge_count=len(edges),
        total_interactions=total_cross,
        density=min(1.0, max(0.0, density)),
        reciprocity=min(1.0, max(0.0, reciprocity)),
        active_pairs=len(edges),
        connected_speakers=connected,
    )
