from app.drift_engine.branch_analysis import (
    BranchDriftDetector,
)
from app.drift_engine.global_drift import (
    GlobalDriftDetector,
)
from app.models.schemas import (
    ThreadData,
    ThreadDriftAnalysis,
)


class DriftEngine:

    def __init__(self) -> None:

        self.global_detector = (
            GlobalDriftDetector()
        )

        self.branch_detector = (
            BranchDriftDetector()
        )

    def analyze(
        self,
        thread: ThreadData,
    ) -> ThreadData:

        global_timeline = (
            self.global_detector.detect(
                thread.nodes
            )
        )

        branches = (
            self.branch_detector.analyze(
                thread.nodes
            )
        )

        thread.drift_analysis = (
            ThreadDriftAnalysis(
                global_timeline=global_timeline,

                branches=branches,

                overall_change_point_count=len(
                    global_timeline.change_points
                ),

                strongest_change_point_id=(
                    global_timeline
                    .strongest_change_point_id
                ),
            )
        )

        # Attach drift intelligence directly to each graph node.
        # The frontend will therefore be able to render the graph without
        # having to reconstruct relationships between nodes and the timeline.
        timeline_by_id = {
            point.comment_id: point
            for point in (
                global_timeline.timeline
            )
        }

        for node in thread.nodes:

            point = timeline_by_id.get(
                node.id
            )

            if point is None:
                continue

            node.features.update(
                {
                    "drift_score": (
                        point.drift_score
                    ),

                    "change_point_probability": (
                        point
                        .change_point_probability
                    ),

                    "recent_change_probability": (
                        point
                        .recent_change_probability
                    ),

                    "run_length_zero_probability": (
                        point
                        .run_length_zero_probability
                    ),

                    "sentiment_delta": (
                        point.sentiment_delta
                    ),

                    "rolling_sentiment": (
                        point.rolling_sentiment
                    ),

                    "toxicity_delta": (
                        point.toxicity_delta
                    ),

                    "phase": point.phase,

                    "is_change_point": (
                        point.is_change_point
                    ),
                }
            )

        return thread