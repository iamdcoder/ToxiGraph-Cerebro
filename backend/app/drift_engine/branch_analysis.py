from app.drift_engine.bocpd import (
    BayesianChangePointDetector,
)
from app.models.schemas import (
    BranchDrift,
    CommentNode,
    ThreadData,
)
from app.services.thread_graph import ThreadGraph


class BranchDriftDetector:

    def __init__(
        self,
        hazard_rate: float = 1 / 50,
        max_run_length: int = 100,
        recent_window: int = 5,
    ) -> None:

        self.hazard_rate = hazard_rate
        self.max_run_length = max_run_length
        self.recent_window = recent_window

    def _get_leaf_paths(
        self,
        graph: ThreadGraph,
    ) -> list[list[str]]:

        paths: list[list[str]] = []

        def walk(
            node_id: str,
            current_path: list[str],
        ) -> None:

            new_path = [
                *current_path,
                node_id,
            ]

            children = graph.children.get(
                node_id,
                [],
            )

            if not children:
                paths.append(new_path)
                return

            for child_id in children:
                walk(
                    child_id,
                    new_path,
                )

        for root_id in graph.get_roots():
            walk(
                root_id,
                [],
            )

        return paths

    def analyze(
        self,
        nodes: list[CommentNode],
    ) -> list[BranchDrift]:

        thread = ThreadData(
            thread_id="branch_analysis",
            platform="custom",
            nodes=nodes,
            edges=[],
        )

        graph = ThreadGraph(thread)
        graph.validate_structure()

        paths = self._get_leaf_paths(
            graph
        )

        node_lookup = {
            node.id: node
            for node in nodes
        }

        results: list[BranchDrift] = []

        for path in paths:

            if len(path) < 3:
                continue

            detector = BayesianChangePointDetector(
                hazard_rate=self.hazard_rate,
                max_run_length=self.max_run_length,
                recent_window=self.recent_window,
            )

            probabilities: list[float] = []

            valid_ids: list[str] = []

            for node_id in path:

                node = node_lookup[node_id]

                if node.analysis is None:
                    continue

                sentiment = (
                    node.analysis
                    .sentiment
                    .sentiment_score
                )

                probability = detector.update(
                    sentiment
                )

                probabilities.append(
                    probability
                )

                valid_ids.append(
                    node_id
                )

            if not probabilities:
                continue

            strongest_index = max(
                range(len(probabilities)),
                key=lambda index: probabilities[index],
            )

            results.append(
                BranchDrift(
                    branch_root_id=path[0],
                    leaf_comment_id=path[-1],
                    path_comment_ids=path,
                    max_change_point_probability=round(
                        probabilities[
                            strongest_index
                        ],
                        4,
                    ),
                    strongest_comment_id=(
                        valid_ids[
                            strongest_index
                        ]
                    ),
                )
            )

        return results