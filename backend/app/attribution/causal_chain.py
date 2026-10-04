from __future__ import annotations

from app.models.schemas import (
    AttributionCandidate,
    CausalChainLink,
    CommentNode,
)


class CausalChainBuilder:
    """
    Build a human-readable path around the strongest contributing trigger.

    Preference:
    1. Use reply topology when the trigger is an ancestor of the fracture.
    2. Otherwise use chronological proximity around the trigger.
    """

    def __init__(
        self,
        max_links: int = 6,
    ) -> None:

        self.max_links = max_links

    @staticmethod
    def _path_to_root(
        node_lookup: dict[str, CommentNode],
        node_id: str,
    ) -> list[str]:

        path: list[str] = []

        current = node_id
        seen: set[str] = set()

        while (
            current in node_lookup
            and current not in seen
        ):

            path.append(current)
            seen.add(current)

            parent_id = (
                node_lookup[current]
                .parent_id
            )

            if parent_id is None:
                break

            current = parent_id

        path.reverse()

        return path

    @staticmethod
    def _is_ancestor(
        node_lookup: dict[str, CommentNode],
        ancestor_id: str,
        descendant_id: str,
    ) -> bool:

        current = descendant_id
        seen: set[str] = set()

        while (
            current in node_lookup
            and current not in seen
        ):

            if current == ancestor_id:
                return True

            seen.add(current)

            parent_id = (
                node_lookup[current]
                .parent_id
            )

            if parent_id is None:
                break

            current = parent_id

        return False

    def _topology_chain(
        self,
        node_lookup: dict[str, CommentNode],
        trigger_id: str,
        fracture_id: str,
    ) -> list[str]:

        if not self._is_ancestor(
            node_lookup,
            trigger_id,
            fracture_id,
        ):
            return []

        path_to_root = self._path_to_root(
            node_lookup,
            fracture_id,
        )

        try:
            start = path_to_root.index(
                trigger_id
            )
        except ValueError:
            return []

        return path_to_root[
            start:
        ]

    def _temporal_chain(
        self,
        nodes: list[CommentNode],
        trigger_id: str,
        fracture_id: str,
    ) -> list[str]:

        ordered = sorted(
            nodes,
            key=lambda node: (
                node.timestamp,
                node.id,
            ),
        )

        ids = [
            node.id
            for node in ordered
        ]

        if trigger_id not in ids:
            return []

        trigger_index = ids.index(
            trigger_id
        )

        fracture_index = ids.index(
            fracture_id
        ) if fracture_id in ids else trigger_index

        if fracture_index < trigger_index:
            return []

        path = ids[
            trigger_index:
            fracture_index + 1
        ]

        if len(path) <= self.max_links + 1:
            return path

        return (
            path[:self.max_links]
            + [fracture_id]
        )

    def build(
        self,
        nodes: list[CommentNode],
        primary: AttributionCandidate,
        fracture_id: str,
    ) -> list[CausalChainLink]:

        node_lookup = {
            node.id: node
            for node in nodes
        }

        topology_path = (
            self._topology_chain(
                node_lookup,
                primary.comment_id,
                fracture_id,
            )
        )

        if topology_path:
            path = topology_path[
                :self.max_links + 1
            ]
            relationship = "reply_chain"

        else:
            path = self._temporal_chain(
                nodes,
                primary.comment_id,
                fracture_id,
            )

            relationship = (
                "temporal_proximity"
            )

        if len(path) < 2:
            return []

        links: list[
            CausalChainLink
        ] = []

        for index in range(
            len(path) - 1
        ):

            source_id = path[index]
            target_id = path[index + 1]

            source = node_lookup[
                source_id
            ]

            target = node_lookup[
                target_id
            ]

            influence = 0.0
            evidence: list[str] = []

            source_features = (
                source.features
            )

            influence += 0.45 * float(
                source_features.get(
                    "structural_impact_score",
                    0.0,
                )
            )

            influence += 0.30 * abs(
                float(
                    source_features.get(
                        "sentiment_delta",
                        0.0,
                    )
                )
            )

            influence += 0.25 * float(
                source_features.get(
                    "toxicity_score",
                    0.0,
                )
            )

            if (
                source_features.get(
                    "is_change_point",
                    False,
                )
            ):
                evidence.append(
                    "Source comment is a detected change point."
                )

            if float(
                source_features.get(
                    "toxicity_score",
                    0.0,
                )
            ) >= 0.50:
                evidence.append(
                    "Source comment carries a strong toxicity signal."
                )

            if (
                float(
                    source_features.get(
                        "sentiment_delta",
                        0.0,
                    )
                )
                <= -0.20
            ):
                evidence.append(
                    "Source comment introduced a sharp negative movement."
                )

            if not evidence:
                evidence.append(
                    "Adjacent comments form the detected escalation path."
                )

            links.append(
                CausalChainLink(
                    from_comment_id=source_id,
                    to_comment_id=target_id,
                    relationship=relationship,
                    influence_score=round(
                        max(
                            0.0,
                            min(
                                influence,
                                1.0,
                            ),
                        ),
                        4,
                    ),
                    evidence=evidence,
                )
            )

        return links