from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from app.services.thread_graph import ThreadGraph


class StructuralFeatureExtractor:
    """
    Extract deterministic graph and topology features for every comment.

    These features describe where a comment sits in the thread and how much
    of the conversation is structurally connected to it.
    """

    def __init__(
        self,
        graph: ThreadGraph,
    ) -> None:
        self.graph = graph
        self.nodes = graph.nodes

        self.ordered_nodes = sorted(
            self.nodes.values(),
            key=lambda node: (
                node.timestamp,
                node.id,
            ),
        )

        self.total_nodes = len(
            self.ordered_nodes
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
    def _safe_ratio(
        numerator: float,
        denominator: float,
    ) -> float:
        if denominator <= 0:
            return 0.0

        return numerator / denominator

    def _subtree_sizes(
        self,
    ) -> dict[str, int]:
        """
        Calculate the number of nodes in each node's subtree,
        including the node itself.
        """

        sizes = {
            node_id: 1
            for node_id in self.nodes
        }

        ordered_by_depth = sorted(
            self.nodes.values(),
            key=lambda node: (
                node.depth,
                node.timestamp,
                node.id,
            ),
            reverse=True,
        )

        for node in ordered_by_depth:
            parent_id = node.parent_id

            if (
                parent_id is not None
                and parent_id in sizes
            ):
                sizes[parent_id] += (
                    sizes[node.id]
                )

        return sizes

    def _branch_roots(
        self,
    ) -> dict[str, str]:
        """
        Map every node to the top-level conversational branch.

        The root comment belongs to itself.
        A direct reply to the root becomes a branch root.
        Every descendant of that reply inherits the same branch id.
        """

        roots = self.graph.get_roots()

        branch_roots: dict[
            str,
            str,
        ] = {}

        for root_id in roots:
            branch_roots[root_id] = root_id

            direct_children = (
                self.graph.children.get(
                    root_id,
                    [],
                )
            )

            for child_id in direct_children:

                stack = [child_id]

                while stack:

                    current = stack.pop()

                    branch_roots[
                        current
                    ] = child_id

                    stack.extend(
                        self.graph.children.get(
                            current,
                            [],
                        )
                    )

        for node_id in self.nodes:

            if node_id in branch_roots:
                continue

            current = node_id
            seen: set[str] = set()

            while (
                current not in seen
            ):

                seen.add(current)

                parent_id = (
                    self.graph.parent.get(
                        current
                    )
                )

                if parent_id is None:
                    branch_roots[
                        node_id
                    ] = current
                    break

                if (
                    parent_id
                    in roots
                ):
                    branch_roots[
                        node_id
                    ] = current
                    break

                current = parent_id

        return branch_roots

    def extract(
        self,
    ) -> dict[str, dict[str, Any]]:
        """
        Extract structural intelligence for every comment.
        """

        self.graph.validate_structure()

        if not self.ordered_nodes:
            return {}

        subtree_sizes = (
            self._subtree_sizes()
        )

        branch_roots = (
            self._branch_roots()
        )

        author_total_counts = Counter(
            node.author
            for node in self.ordered_nodes
        )

        author_seen_counts: Counter[str] = (
            Counter()
        )

        root_timestamps = [
            self.nodes[root_id].timestamp
            for root_id in self.graph.get_roots()
        ]

        thread_start = min(
            root_timestamps
        )

        thread_end = max(
            node.timestamp
            for node in self.ordered_nodes
        )

        thread_duration = max(
            (
                thread_end
                - thread_start
            ).total_seconds(),
            0.0,
        )

        reply_velocities = [
            max(
                (
                    node.timestamp
                    - self.nodes[
                        node.parent_id
                    ].timestamp
                ).total_seconds(),
                0.0,
            )
            for node in self.ordered_nodes
            if node.parent_id is not None
        ]

        velocity_reference = (
            median(reply_velocities)
            if reply_velocities
            else 0.0
        )

        velocity_scale = max(
            velocity_reference,
            1.0,
        )

        max_depth = max(
            (
                node.depth
                for node in self.ordered_nodes
            ),
            default=0,
        )

        max_branching = max(
            (
                len(
                    self.graph.children.get(
                        node.id,
                        [],
                    )
                )
                for node in self.ordered_nodes
            ),
            default=0,
        )

        features: dict[
            str,
            dict[str, Any],
        ] = {}

        for index, node in enumerate(
            self.ordered_nodes
        ):
            parent = (
                self.nodes[node.parent_id]
                if node.parent_id is not None
                else None
            )

            children = (
                self.graph.children.get(
                    node.id,
                    [],
                )
            )

            reply_velocity = 0.0

            if parent is not None:
                reply_velocity = max(
                    (
                        node.timestamp
                        - parent.timestamp
                    ).total_seconds(),
                    0.0,
                )

            time_since_start = max(
                (
                    node.timestamp
                    - thread_start
                ).total_seconds(),
                0.0,
            )

            author_seen_counts[
                node.author
            ] += 1

            total_siblings = (
                len(
                    self.graph.children.get(
                        node.parent_id,
                        [],
                    )
                )
                if node.parent_id is not None
                else 0
            )

            sibling_count = max(
                total_siblings - 1,
                0,
            )

            branch_root_id = (
                branch_roots.get(
                    node.id,
                    node.id,
                )
            )

            depth_ratio = self._safe_ratio(
                node.depth,
                max_depth,
            )

            branching_factor = len(
                children
            )

            branching_score = (
                self._safe_ratio(
                    branching_factor,
                    max_branching,
                )
            )

            subtree_share = (
                self._safe_ratio(
                    subtree_sizes.get(
                        node.id,
                        1,
                    ),
                    self.total_nodes,
                )
            )

            sibling_ratio = (
                self._safe_ratio(
                    sibling_count,
                    total_siblings,
                )
                if node.parent_id is not None
                else 0.0
            )

            structural_impact_score = (
                self._clamp(
                    0.40 * subtree_share
                    + 0.25 * branching_score
                    + 0.20 * depth_ratio
                    + 0.15 * sibling_ratio
                )
            )

            reply_velocity_score = (
                0.0
                if parent is None
                else 1.0
                / (
                    1.0
                    + reply_velocity
                    / velocity_scale
                )
            )

            is_cross_reply = bool(
                parent is not None
                and node.author != parent.author
            )

            author_reentry_count = max(
                author_seen_counts[
                    node.author
                ]
                - 1,
                0,
            )

            thread_position = (
                self._safe_ratio(
                    index,
                    max(
                        self.total_nodes - 1,
                        1,
                    ),
                )
            )

            positional_ratio = (
                self._safe_ratio(
                    index + 1,
                    self.total_nodes,
                )
            )

            features[node.id] = {
                "chronological_index": index,

                "depth": node.depth,

                "depth_ratio": round(
                    depth_ratio,
                    4,
                ),

                "branch_root_id": (
                    branch_root_id
                ),

                "branching_factor": (
                    branching_factor
                ),

                "branching_score": round(
                    branching_score,
                    4,
                ),

                "subtree_size": (
                    subtree_sizes.get(
                        node.id,
                        1,
                    )
                ),

                "subtree_share": round(
                    subtree_share,
                    4,
                ),

                "sibling_count": (
                    sibling_count
                ),

                "sibling_ratio": round(
                    sibling_ratio,
                    4,
                ),

                "author_comment_count": (
                    author_total_counts[
                        node.author
                    ]
                ),

                "author_reentry_count": (
                    author_reentry_count
                ),

                "reply_velocity": round(
                    reply_velocity,
                    4,
                ),

                "reply_velocity_score": round(
                    reply_velocity_score,
                    4,
                ),

                "time_since_parent": round(
                    reply_velocity,
                    4,
                ),

                "time_since_thread_start": round(
                    time_since_start,
                    4,
                ),

                "thread_position": round(
                    thread_position,
                    4,
                ),

                "positional_ratio": round(
                    positional_ratio,
                    4,
                ),

                "thread_duration": round(
                    thread_duration,
                    4,
                ),

                "is_cross_reply": (
                    is_cross_reply
                ),

                "structural_impact_score": round(
                    structural_impact_score,
                    4,
                ),
            }

        return features