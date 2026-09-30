from __future__ import annotations

from collections import defaultdict, deque

from app.models.schemas import (
    CommentNode,
    ThreadData,
    ThreadEdge,
)


class ThreadGraph:
    """
    Validated reply graph for a single conversation thread.
    """

    def __init__(
        self,
        thread: ThreadData,
    ):
        self.thread = thread

        self.nodes: dict[
            str,
            CommentNode,
        ] = {
            node.id: node
            for node in thread.nodes
        }

        self.children: dict[
            str,
            list[str],
        ] = defaultdict(list)

        self.parent: dict[
            str,
            str | None,
        ] = {}

        self.edges: list[
            ThreadEdge
        ] = []

        self._build()

    def _build(self) -> None:
        for node in self.thread.nodes:
            self.parent[node.id] = (
                node.parent_id
            )

            if node.parent_id is not None:
                if (
                    node.parent_id
                    not in self.nodes
                ):
                    raise ValueError(
                        f"Parent comment "
                        f"'{node.parent_id}' "
                        f"does not exist for "
                        f"'{node.id}'."
                    )

                self.children[
                    node.parent_id
                ].append(node.id)

        # Keep child ordering deterministic even when the source JSON
        # is shuffled. This makes branch analysis and graph rendering stable.
        for (
            parent_id,
            child_ids,
        ) in self.children.items():

            child_ids.sort(
                key=lambda child_id: (
                    self.nodes[
                        child_id
                    ].timestamp,
                    child_id,
                )
            )

            parent_node = (
                self.nodes[parent_id]
            )

            for child_id in child_ids:
                child_node = (
                    self.nodes[child_id]
                )

                time_delta = max(
                    (
                        child_node.timestamp
                        - parent_node.timestamp
                    ).total_seconds(),
                    0.0,
                )

                self.edges.append(
                    ThreadEdge(
                        source=parent_id,
                        target=child_id,
                        edge_type="reply_to",
                        time_delta_seconds=(
                            time_delta
                        ),
                    )
                )

        self.edges.sort(
            key=lambda edge: (
                self.nodes[
                    edge.target
                ].timestamp,
                edge.source,
                edge.target,
            )
        )

    def get_roots(self) -> list[str]:
        roots = [
            node_id
            for (
                node_id,
                parent_id,
            ) in self.parent.items()
            if parent_id is None
        ]

        return sorted(
            roots,
            key=lambda node_id: (
                self.nodes[
                    node_id
                ].timestamp,
                node_id,
            ),
        )

    def calculate_depths(self) -> None:
        queue = deque(
            (
                root_id,
                0,
            )
            for root_id in self.get_roots()
        )

        visited: set[str] = set()

        while queue:
            (
                node_id,
                depth,
            ) = queue.popleft()

            if node_id in visited:
                continue

            visited.add(node_id)

            node = self.nodes[
                node_id
            ]

            node.depth = depth
            node.is_op = depth == 0

            node.children_count = len(
                self.children.get(
                    node_id,
                    [],
                )
            )

            for child_id in self.children.get(
                node_id,
                [],
            ):
                queue.append(
                    (
                        child_id,
                        depth + 1,
                    )
                )

    def validate_structure(self) -> None:
        roots = self.get_roots()

        if not roots:
            raise ValueError(
                "Thread must contain at least "
                "one root comment."
            )

        self.calculate_depths()

        visited: set[str] = set()
        active: set[str] = set()

        def dfs(node_id: str) -> None:
            if node_id in active:
                raise ValueError(
                    "Cycle detected in "
                    "thread graph."
                )

            if node_id in visited:
                return

            active.add(node_id)

            for child_id in self.children.get(
                node_id,
                [],
            ):
                dfs(child_id)

            active.remove(node_id)
            visited.add(node_id)

        for root_id in roots:
            dfs(root_id)

        if len(visited) != len(
            self.nodes
        ):
            raise ValueError(
                "Some comments are disconnected "
                "from the thread roots."
            )

    def to_thread_data(
        self,
    ) -> ThreadData:
        self.validate_structure()

        ordered_nodes = sorted(
            self.nodes.values(),
            key=lambda node: (
                node.timestamp,
                node.id,
            ),
        )

        return ThreadData(
            thread_id=self.thread.thread_id,
            platform=self.thread.platform,
            title=self.thread.title,
            created_at=self.thread.created_at,
            nodes=ordered_nodes,
            edges=list(self.edges),
        )

    def summary(self) -> dict:
        self.validate_structure()

        max_depth = max(
            (
                node.depth
                for node in self.nodes.values()
            ),
            default=0,
        )

        return {
            "thread_id": (
                self.thread.thread_id
            ),
            "platform": (
                self.thread.platform
            ),
            "comment_count": (
                len(self.nodes)
            ),
            "edge_count": (
                len(self.edges)
            ),
            "root_count": (
                len(self.get_roots())
            ),
            "max_depth": max_depth,
        }