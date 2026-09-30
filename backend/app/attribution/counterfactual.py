from __future__ import annotations

from copy import deepcopy

from app.drift_engine.engine import DriftEngine
from app.models.schemas import (
    AttributionCandidate,
    CommentNode,
    ThreadData,
)


class CounterfactualAttributor:
    """
    Estimate which nearby comments contributed most strongly to a detected
    fracture point.

    Candidate selection is topology-aware:
    - same branch as the fracture,
    - ancestors of the fracture,
    - or immediately preceding comments on that branch.

    The attribution is a contribution estimate, not proof of intent or
    definitive causation.
    """

    def __init__(
        self,
        candidate_window: int = 5,
        max_candidates: int = 5,
    ) -> None:

        self.candidate_window = (
            candidate_window
        )

        self.max_candidates = (
            max_candidates
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:

        return max(
            minimum,
            min(
                value,
                maximum,
            ),
        )

    @staticmethod
    def _comment_by_id(
        nodes: list[CommentNode],
    ) -> dict[str, CommentNode]:

        return {
            node.id: node
            for node in nodes
        }

    @staticmethod
    def _path_to_root(
        lookup: dict[str, CommentNode],
        node_id: str,
    ) -> list[str]:

        path: list[str] = []

        current = node_id
        seen: set[str] = set()

        while (
            current in lookup
            and current not in seen
        ):

            path.append(
                current
            )

            seen.add(
                current
            )

            parent_id = (
                lookup[current]
                .parent_id
            )

            if parent_id is None:
                break

            current = parent_id

        path.reverse()

        return path

    def _local_event_strength(
        self,
        thread: ThreadData,
        neighborhood_ids: set[str],
    ) -> tuple[float, float]:

        drift_analysis = (
            thread.drift_analysis
        )

        if drift_analysis is None:
            return 0.0, 0.0

        points = [
            point
            for point in (
                drift_analysis
                .global_timeline
                .timeline
            )
            if point.comment_id
            in neighborhood_ids
        ]

        if not points:
            return 0.0, 0.0

        strongest = max(
            points,
            key=lambda point: (
                point.drift_score,
                point.change_point_probability,
            ),
        )

        return (
            strongest.drift_score,
            strongest.change_point_probability,
        )

    def _remove_candidate(
        self,
        thread: ThreadData,
        candidate_id: str,
    ) -> ThreadData:

        modified = deepcopy(
            thread
        )

        candidate = next(
            (
                node
                for node in modified.nodes
                if node.id == candidate_id
            ),
            None,
        )

        if candidate is None:
            return modified

        replacement_parent_id = (
            candidate.parent_id
        )

        for node in modified.nodes:

            if node.parent_id == candidate_id:
                node.parent_id = (
                    replacement_parent_id
                )

        modified.nodes = [
            node
            for node in modified.nodes
            if node.id != candidate_id
        ]

        modified.edges = []

        return modified

    def _build_evidence(
        self,
        node: CommentNode,
    ) -> list[str]:

        evidence: list[str] = []

        features = node.features
        analysis = node.analysis

        if analysis is None:
            return evidence

        sentiment_delta = float(
            features.get(
                "sentiment_delta",
                0.0,
            )
        )

        toxicity_delta = float(
            features.get(
                "toxicity_delta",
                0.0,
            )
        )

        pronoun_shift = float(
            analysis
            .linguistic
            .pronoun_shift
        )

        negation_density = float(
            analysis
            .linguistic
            .negation_density
        )

        hedging_score = float(
            analysis
            .linguistic
            .hedging_score
        )

        reply_velocity_score = float(
            features.get(
                "reply_velocity_score",
                0.0,
            )
        )

        profanity_score = float(
            analysis
            .linguistic
            .profanity_score
        )

        toxicity = float(
            analysis
            .toxicity
            .toxicity
        )

        if sentiment_delta <= -0.20:
            evidence.append(
                "Sentiment dropped sharply at this reply."
            )

        if toxicity_delta >= 0.10:
            evidence.append(
                "Toxicity increased immediately."
            )

        if toxicity >= 0.50:
            evidence.append(
                "The reply carries a strong toxicity signal."
            )

        if pronoun_shift >= 0.50:
            evidence.append(
                "Accusatory 'you' language increased."
            )

        if negation_density >= 0.15:
            evidence.append(
                "Negation/disagreement language is elevated."
            )

        if hedging_score <= 0.05:
            evidence.append(
                "Hedging is low, indicating more categorical language."
            )

        if reply_velocity_score >= 0.65:
            evidence.append(
                "The reply arrived with accelerated response timing."
            )

        if profanity_score > 0:
            evidence.append(
                "Profanity was detected."
            )

        if not evidence:
            evidence.append(
                "The comment is structurally and temporally close to the fracture."
            )

        return evidence

    def _candidate_nodes(
        self,
        nodes: list[CommentNode],
        fracture_index: int,
        fracture_id: str,
    ) -> list[CommentNode]:

        ordered = sorted(
            nodes,
            key=lambda node: (
                node.timestamp,
                node.id,
            ),
        )

        lookup = {
            node.id: node
            for node in ordered
        }

        fracture = lookup[
            fracture_id
        ]

        fracture_branch = (
            fracture
            .features
            .get(
                "branch_root_id"
            )
        )

        ancestors = set(
            self._path_to_root(
                lookup,
                fracture_id,
            )
        )

        ancestors.discard(
            fracture_id
        )

        start = max(
            0,
            fracture_index
            - self.candidate_window,
        )

        nearby = ordered[
            start:fracture_index
        ]

        same_branch = [
            node
            for node in nearby
            if (
                node.id in ancestors
                or
                node.features.get(
                    "branch_root_id"
                )
                == fracture_branch
            )
        ]

        unique: dict[
            str,
            CommentNode,
        ] = {}

        for node in same_branch:
            unique[
                node.id
            ] = node

        return list(
            unique.values()
        )

    def rank_candidates(
        self,
        thread: ThreadData,
    ) -> list[AttributionCandidate]:

        drift_analysis = (
            thread.drift_analysis
        )

        if drift_analysis is None:
            return []

        fracture = (
            drift_analysis
            .global_timeline
            .fracture_point
        )

        if fracture is None:
            return []

        nodes = sorted(
            thread.nodes,
            key=lambda node: (
                node.timestamp,
                node.id,
            ),
        )

        candidate_nodes = (
            self._candidate_nodes(
                nodes,
                fracture
                .chronological_index,
                fracture
                .comment_id,
            )
        )

        if not candidate_nodes:
            return []

        neighborhood_ids = {
            node.id
            for node in nodes[
                max(
                    0,
                    fracture
                    .chronological_index
                    - self.candidate_window,
                ):min(
                    len(nodes),
                    fracture
                    .chronological_index
                    + 3,
                )
            ]
        }

        baseline_drift, baseline_cp = (
            self._local_event_strength(
                thread,
                neighborhood_ids,
            )
        )

        baseline_strength = max(
            baseline_drift,
            fracture.drift_score,
        )

        baseline_cp = max(
            baseline_cp,
            fracture
            .change_point_probability,
        )

        results: list[
            AttributionCandidate
        ] = []

        for node in candidate_nodes:

            modified = (
                self._remove_candidate(
                    thread,
                    node.id,
                )
            )

            try:
                modified = (
                    DriftEngine().analyze(
                        modified
                    )
                )
            except Exception:
                continue

            counterfactual_drift, (
                counterfactual_cp
            ) = self._local_event_strength(
                modified,
                neighborhood_ids
                - {node.id},
            )

            drift_reduction = max(
                baseline_strength
                - counterfactual_drift,
                0.0,
            )

            cp_reduction = max(
                baseline_cp
                - counterfactual_cp,
                0.0,
            )

            normalized_drift_reduction = (
                self._clamp(
                    drift_reduction
                    / max(
                        baseline_strength,
                        0.01,
                    )
                )
            )

            normalized_cp_reduction = (
                self._clamp(
                    cp_reduction
                    / max(
                        baseline_cp,
                        0.01,
                    )
                )
            )

            features = node.features

            local_signal = 0.0

            if (
                float(
                    features.get(
                        "sentiment_delta",
                        0.0,
                    )
                )
                <= -0.20
            ):
                local_signal += 0.30

            if (
                float(
                    features.get(
                        "toxicity_delta",
                        0.0,
                    )
                )
                >= 0.10
            ):
                local_signal += 0.25

            if (
                float(
                    features.get(
                        "pronoun_shift",
                        (
                            node.analysis
                            .linguistic
                            .pronoun_shift
                            if node.analysis is not None
                            else 0.0
                        ),
                    )
                )
                >= 0.50
            ):
                local_signal += 0.15

            if (
                float(
                    features.get(
                        "profanity_score",
                        (
                            node.analysis
                            .linguistic
                            .profanity_score
                            if node.analysis is not None
                            else 0.0
                        ),
                    )
                )
                > 0
            ):
                local_signal += 0.10

            if (
                float(
                    features.get(
                        "reply_velocity_score",
                        0.0,
                    )
                )
                >= 0.65
            ):
                local_signal += 0.10

            if node.analysis is not None:
                if (
                    node.analysis
                    .toxicity
                    .toxicity
                    >= 0.50
                ):
                    local_signal += 0.10

            local_signal = self._clamp(
                local_signal
            )

            contribution_score = (
                0.60
                * normalized_drift_reduction

                + 0.30
                * normalized_cp_reduction

                + 0.10
                * local_signal
            )

            # Zero-effect counterfactuals are not presented as meaningful
            # trigger candidates merely because their wording looks heated.
            if (
                drift_reduction <= 0.001
                and cp_reduction <= 0.001
                and local_signal < 0.40
            ):
                contribution_score = 0.0

            results.append(
                AttributionCandidate(
                    comment_id=node.id,

                    rank=0,

                    contribution_score=round(
                        contribution_score,
                        4,
                    ),

                    baseline_drift_score=round(
                        baseline_strength,
                        4,
                    ),

                    counterfactual_drift_score=round(
                        counterfactual_drift,
                        4,
                    ),

                    drift_reduction=round(
                        drift_reduction,
                        4,
                    ),

                    baseline_change_probability=round(
                        baseline_cp,
                        4,
                    ),

                    counterfactual_change_probability=round(
                        counterfactual_cp,
                        4,
                    ),

                    sentiment_score=round(
                        (
                            node.analysis
                            .sentiment
                            .sentiment_score
                            if node.analysis is not None
                            else 0.0
                        ),
                        4,
                    ),

                    toxicity_score=round(
                        (
                            node.analysis
                            .toxicity
                            .toxicity
                            if node.analysis is not None
                            else 0.0
                        ),
                        4,
                    ),

                    evidence=(
                        self._build_evidence(
                            node
                        )
                    ),
                )
            )

        results.sort(
            key=lambda candidate: (
                candidate.contribution_score,
                candidate.drift_reduction,
                candidate.baseline_change_probability,
            ),
            reverse=True,
        )

        results = results[
            :self.max_candidates
        ]

        ranked: list[
            AttributionCandidate
        ] = []

        for rank, candidate in enumerate(
            results,
            start=1,
        ):
            candidate.rank = rank
            ranked.append(
                candidate
            )

        return ranked