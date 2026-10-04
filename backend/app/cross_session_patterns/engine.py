from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np

from app.longitudinal_emotion.service import HistorySession


class CrossSessionPatternError(ValueError):
    """Raised when cross-session patterns cannot be computed."""


MIN_PATTERN_SESSIONS = 4

AFFECT_KEYS = ("valence", "arousal", "dominance")
ACOUSTIC_KEYS = (
    "pitch_mean_hz",
    "pitch_std_hz",
    "energy_mean",
    "estimated_syllable_rate_sps",
    "estimated_pause_ratio",
    "spectral_centroid_mean_hz",
)

PATTERN_TYPE = {
    "emotion_recurrence": "Emotion recurrence",
    "emotion_persistence": "Emotion persistence",
    "emotion_transition": "Repeated emotion transition",
    "affect_momentum": "Affect direction recurrence",
    "affect_stability": "Affect stability",
    "affect_volatility": "Affect volatility",
    "acoustic_affect_link": "Acoustic–affect association",
}


@dataclass(frozen=True)
class PatternEvidence:
    kind: str
    value: float | None
    unit: str


@dataclass(frozen=True)
class RecurringPattern:
    pattern_type: str
    title: str
    description: str
    strength: float
    confidence: float
    occurrences: int
    evidence: list[PatternEvidence]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "title": self.title,
            "description": self.description,
            "strength": round(self.strength, 4),
            "confidence": round(self.confidence, 4),
            "occurrences": self.occurrences,
            "evidence": [asdict(item) for item in self.evidence],
        }


@dataclass(frozen=True)
class TransitionPattern:
    from_emotion: str
    to_emotion: str
    occurrences: int
    share_of_transitions: float
    strength: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_emotion": self.from_emotion,
            "to_emotion": self.to_emotion,
            "occurrences": self.occurrences,
            "share_of_transitions": round(self.share_of_transitions, 4),
            "strength": round(self.strength, 4),
        }


@dataclass(frozen=True)
class CrossSessionPatternReport:
    available: bool
    profile_id: str
    session_count: int
    labeled_session_count: int
    usable_affect_transitions: int
    patterns: list[RecurringPattern]
    repeated_transitions: list[TransitionPattern]
    dominant_emotion: str | None
    dominant_emotion_share: float | None
    emotion_entropy: float | None
    recurrence_index: float
    summary: str
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "profile_id": self.profile_id,
            "session_count": self.session_count,
            "labeled_session_count": self.labeled_session_count,
            "usable_affect_transitions": self.usable_affect_transitions,
            "patterns": [item.to_dict() for item in self.patterns],
            "repeated_transitions": [item.to_dict() for item in self.repeated_transitions],
            "dominant_emotion": self.dominant_emotion,
            "dominant_emotion_share": None if self.dominant_emotion_share is None else round(self.dominant_emotion_share, 4),
            "emotion_entropy": None if self.emotion_entropy is None else round(self.emotion_entropy, 4),
            "recurrence_index": round(self.recurrence_index, 4),
            "summary": self.summary,
            "caveat": self.caveat,
        }


class CrossSessionPatternEngine:
    def analyze(self, profile_id: str, sessions: Iterable[HistorySession]) -> CrossSessionPatternReport:
        ordered = sorted(list(sessions), key=lambda item: item.recorded_at)
        session_count = len(ordered)
        if session_count < MIN_PATTERN_SESSIONS:
            return self._insufficient(profile_id, session_count)

        labeled = [item for item in ordered if item.emotion]
        emotion_counts: dict[str, int] = {}
        for item in labeled:
            emotion_counts[item.emotion] = emotion_counts.get(item.emotion, 0) + 1

        dominant_emotion = None
        dominant_share = None
        entropy = None
        if labeled:
            dominant_emotion, dominant_count = max(
                emotion_counts.items(), key=lambda pair: (pair[1], pair[0])
            )
            dominant_share = dominant_count / len(labeled)
            probabilities = np.asarray(
                [count / len(labeled) for count in emotion_counts.values()], dtype=np.float64
            )
            entropy = float(
                -np.sum(probabilities * np.log(probabilities + 1e-12))
                / max(math.log(max(len(emotion_counts), 2)), 1e-12)
            ) if len(emotion_counts) > 1 else 0.0

        transitions = self._emotion_transitions(labeled)
        patterns: list[RecurringPattern] = []
        patterns.extend(self._emotion_recurrence_patterns(labeled, dominant_emotion, dominant_share))
        patterns.extend(self._emotion_persistence_patterns(labeled))
        patterns.extend(self._affect_momentum_patterns(ordered))
        patterns.extend(self._affect_stability_patterns(ordered))
        patterns.extend(self._acoustic_affect_patterns(ordered))

        repeated = [
            item
            for item in transitions
            if item.occurrences >= 2 and item.share_of_transitions >= 0.15
        ]
        for item in repeated[:4]:
            confidence = _sample_confidence(len(labeled), item.occurrences, len(transitions))
            patterns.append(
                RecurringPattern(
                    pattern_type="emotion_transition",
                    title=f"{item.from_emotion} → {item.to_emotion} recurs",
                    description=(
                        f"The sequence {item.from_emotion} → {item.to_emotion} appeared "
                        f"{item.occurrences} times across consecutive labeled sessions."
                    ),
                    strength=item.strength,
                    confidence=confidence,
                    occurrences=item.occurrences,
                    evidence=[
                        PatternEvidence("transition_share", item.share_of_transitions, "fraction"),
                        PatternEvidence("transition_count", float(item.occurrences), "sessions"),
                    ],
                )
            )

        patterns = sorted(patterns, key=lambda item: (item.strength * item.confidence, item.occurrences), reverse=True)[:8]
        recurrence_index = self._recurrence_index(labeled, repeated, patterns)
        summary = self._summary(
            session_count=session_count,
            labeled_count=len(labeled),
            dominant_emotion=dominant_emotion,
            dominant_share=dominant_share,
            repeated=repeated,
            pattern_count=len(patterns),
        )

        return CrossSessionPatternReport(
            available=True,
            profile_id=profile_id,
            session_count=session_count,
            labeled_session_count=len(labeled),
            usable_affect_transitions=self._usable_affect_transitions(ordered),
            patterns=patterns,
            repeated_transitions=sorted(repeated, key=lambda item: item.strength, reverse=True)[:8],
            dominant_emotion=dominant_emotion,
            dominant_emotion_share=dominant_share,
            emotion_entropy=entropy,
            recurrence_index=recurrence_index,
            summary=summary,
            caveat=(
                "Patterns describe recurrence in recorded vocal behavior and affect summaries across saved sessions. "
                "They are descriptive associations, not measurements of private emotional state, health, or causation."
            ),
        )

    def _emotion_recurrence_patterns(
        self,
        labeled: list[HistorySession],
        dominant_emotion: str | None,
        dominant_share: float | None,
    ) -> list[RecurringPattern]:
        if not labeled or dominant_emotion is None or dominant_share is None or dominant_share < 0.35:
            return []
        count = sum(item.emotion == dominant_emotion for item in labeled)
        strength = float(np.clip(0.5 * dominant_share + 0.5 * min(count / 6.0, 1.0), 0.0, 1.0))
        confidence = _sample_confidence(len(labeled), count, len(labeled))
        return [
            RecurringPattern(
                pattern_type="emotion_recurrence",
                title=f"{dominant_emotion} is the most recurrent labeled emotion",
                description=(
                    f"{dominant_emotion} appears in {count} of {len(labeled)} labeled sessions "
                    f"({dominant_share * 100:.0f}%)."
                ),
                strength=strength,
                confidence=confidence,
                occurrences=count,
                evidence=[PatternEvidence("session_share", dominant_share, "fraction")],
            )
        ]

    def _emotion_persistence_patterns(self, labeled: list[HistorySession]) -> list[RecurringPattern]:
        if len(labeled) < 3:
            return []
        runs: list[int] = []
        current = 1
        for previous, item in zip(labeled, labeled[1:]):
            if item.emotion == previous.emotion:
                current += 1
            else:
                runs.append(current)
                current = 1
        runs.append(current)
        repeated_runs = sum(run for run in runs if run >= 2)
        max_run = max(runs)
        if repeated_runs < 2 or max_run < 2:
            return []
        strength = float(np.clip(0.55 * (repeated_runs / len(labeled)) + 0.45 * min(max_run / 5, 1), 0.0, 1.0))
        confidence = _sample_confidence(len(labeled), repeated_runs, len(runs))
        return [
            RecurringPattern(
                pattern_type="emotion_persistence",
                title="Repeated same-emotion runs recur",
                description=(
                    f"Consecutive labeled sessions share the same emotion in {sum(1 for run in runs if run >= 2)} "
                    f"runs; the longest run is {max_run} sessions."
                ),
                strength=strength,
                confidence=confidence,
                occurrences=sum(1 for run in runs if run >= 2),
                evidence=[PatternEvidence("longest_run", float(max_run), "sessions")],
            )
        ]

    def _affect_momentum_patterns(self, sessions: list[HistorySession]) -> list[RecurringPattern]:
        output: list[RecurringPattern] = []
        for key, label in (("valence", "Valence"), ("arousal", "Arousal"), ("dominance", "Dominance")):
            deltas: list[float] = []
            for previous, current in zip(sessions, sessions[1:]):
                a = previous.affect.get(key)
                b = current.affect.get(key)
                if a is None or b is None:
                    continue
                delta = float(b) - float(a)
                if abs(delta) >= 0.035:
                    deltas.append(delta)
            if len(deltas) < 3:
                continue
            positive = sum(item > 0 for item in deltas)
            negative = sum(item < 0 for item in deltas)
            dominant_direction = "rises" if positive >= negative else "falls"
            count = max(positive, negative)
            share = count / len(deltas)
            if share < 0.6:
                continue
            strength = float(np.clip(share * min(len(deltas) / 8.0, 1.0), 0.0, 1.0))
            confidence = _sample_confidence(len(sessions), count, len(deltas))
            output.append(
                RecurringPattern(
                    pattern_type="affect_momentum",
                    title=f"{label} repeatedly {dominant_direction} between sessions",
                    description=(
                        f"Among {len(deltas)} meaningful session-to-session changes, {count} "
                        f"({share * 100:.0f}%) move in the same direction."
                    ),
                    strength=strength,
                    confidence=confidence,
                    occurrences=count,
                    evidence=[PatternEvidence("direction_share", share, "fraction")],
                )
            )
        return output

    def _affect_stability_patterns(self, sessions: list[HistorySession]) -> list[RecurringPattern]:
        vectors: list[list[float]] = []
        for item in sessions:
            values = [item.affect.get(key) for key in AFFECT_KEYS]
            if all(value is not None for value in values):
                vectors.append([float(value) for value in values])
        if len(vectors) < 4:
            return []
        distances = [float(np.linalg.norm(np.asarray(b) - np.asarray(a))) for a, b in zip(vectors, vectors[1:])]
        mean_distance = float(np.mean(distances))
        normalized = float(np.clip(mean_distance / 0.85, 0.0, 1.0))
        if normalized < 0.2:
            kind = "affect_stability"
            title = "Affect remains comparatively stable"
            description = f"Mean V/A/D movement between usable sessions is {mean_distance:.3f}."
            strength = 1.0 - normalized
        elif normalized > 0.48:
            kind = "affect_volatility"
            title = "Affect shows recurring session-to-session volatility"
            description = f"Mean V/A/D movement between usable sessions is {mean_distance:.3f}."
            strength = normalized
        else:
            return []
        return [
            RecurringPattern(
                pattern_type=kind,
                title=title,
                description=description,
                strength=float(np.clip(strength, 0.0, 1.0)),
                confidence=_sample_confidence(len(vectors), len(distances), len(vectors)),
                occurrences=len(distances),
                evidence=[PatternEvidence("mean_affect_distance", mean_distance, "VAD units")],
            )
        ]

    def _acoustic_affect_patterns(self, sessions: list[HistorySession]) -> list[RecurringPattern]:
        if len(sessions) < 5:
            return []
        output: list[RecurringPattern] = []
        for acoustic_key, affect_key in (("energy_mean", "arousal"), ("pitch_mean_hz", "arousal"), ("estimated_syllable_rate_sps", "arousal")):
            pairs: list[tuple[float, float]] = []
            for item in sessions:
                acoustic = item.acoustic.get(acoustic_key)
                affect = item.affect.get(affect_key)
                if acoustic is None or affect is None:
                    continue
                pairs.append((float(acoustic), float(affect)))
            if len(pairs) < 5:
                continue
            x, y = np.asarray(pairs, dtype=np.float64).T
            if float(np.std(x)) < 1e-9 or float(np.std(y)) < 1e-9:
                continue
            correlation = float(np.corrcoef(x, y)[0, 1])
            if not math.isfinite(correlation) or abs(correlation) < 0.55:
                continue
            direction = "positive" if correlation > 0 else "negative"
            label = acoustic_key.replace("_", " ")
            output.append(
                RecurringPattern(
                    pattern_type="acoustic_affect_link",
                    title=f"{label} tracks arousal ({direction} association)",
                    description=(
                        f"Across {len(pairs)} usable sessions, the Pearson correlation between "
                        f"{label} and arousal is {correlation:+.2f}."
                    ),
                    strength=float(np.clip(abs(correlation), 0.0, 1.0)),
                    confidence=_sample_confidence(len(pairs), len(pairs), len(pairs)) * 0.9,
                    occurrences=len(pairs),
                    evidence=[PatternEvidence("pearson_r", correlation, "correlation")],
                )
            )
        return output

    def _emotion_transitions(self, labeled: list[HistorySession]) -> list[TransitionPattern]:
        counts: dict[tuple[str, str], int] = {}
        for previous, current in zip(labeled, labeled[1:]):
            key = (previous.emotion or "", current.emotion or "")
            counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        if total == 0:
            return []
        return sorted(
            [
                TransitionPattern(
                    from_emotion=source,
                    to_emotion=target,
                    occurrences=count,
                    share_of_transitions=count / total,
                    strength=float(np.clip((count / total) * min(count / 3.0, 1.0), 0.0, 1.0)),
                )
                for (source, target), count in counts.items()
            ],
            key=lambda item: (item.occurrences, item.share_of_transitions),
            reverse=True,
        )

    def _usable_affect_transitions(self, sessions: list[HistorySession]) -> int:
        total = 0
        for previous, current in zip(sessions, sessions[1:]):
            if all(previous.affect.get(key) is not None and current.affect.get(key) is not None for key in AFFECT_KEYS):
                total += 1
        return total

    def _recurrence_index(
        self,
        labeled: list[HistorySession],
        repeated: list[TransitionPattern],
        patterns: list[RecurringPattern],
    ) -> float:
        transition_component = 0.0
        if len(labeled) > 1:
            transition_component = min(sum(item.occurrences for item in repeated) / (len(labeled) - 1), 1.0)
        pattern_component = min(len(patterns) / 5.0, 1.0)
        return float(np.clip(0.65 * transition_component + 0.35 * pattern_component, 0.0, 1.0))

    def _summary(
        self,
        session_count: int,
        labeled_count: int,
        dominant_emotion: str | None,
        dominant_share: float | None,
        repeated: list[TransitionPattern],
        pattern_count: int,
    ) -> str:
        if pattern_count == 0:
            return (
                f"CEREBRO found no strong recurring cross-session pattern across {session_count} sessions "
                "under the current minimum-evidence rules."
            )
        lead = f"CEREBRO found {pattern_count} recurring pattern{'s' if pattern_count != 1 else ''} across {session_count} sessions."
        if dominant_emotion and dominant_share is not None:
            lead += f" {dominant_emotion} is the most common labeled emotion at {dominant_share * 100:.0f}%."
        if repeated:
            transition = repeated[0]
            lead += f" The most repeated transition is {transition.from_emotion} → {transition.to_emotion}."
        return lead

    def _insufficient(self, profile_id: str, session_count: int) -> CrossSessionPatternReport:
        return CrossSessionPatternReport(
            available=False,
            profile_id=profile_id,
            session_count=session_count,
            labeled_session_count=0,
            usable_affect_transitions=0,
            patterns=[],
            repeated_transitions=[],
            dominant_emotion=None,
            dominant_emotion_share=None,
            emotion_entropy=None,
            recurrence_index=0.0,
            summary=(
                f"At least {MIN_PATTERN_SESSIONS} saved sessions are needed before CEREBRO can search for recurring patterns."
            ),
            caveat="CEREBRO intentionally avoids declaring recurring patterns from very small histories.",
        )


def _sample_confidence(sample_count: int, occurrences: int, denominator: int) -> float:
    if sample_count <= 0 or denominator <= 0:
        return 0.0
    sample_factor = min(1.0, 0.35 + 0.08 * min(sample_count, 8))
    recurrence_factor = min(1.0, occurrences / max(2.0, denominator * 0.4))
    return float(np.clip(sample_factor * (0.55 + 0.45 * recurrence_factor), 0.0, 1.0))
