from datetime import (
    datetime,
    timedelta,
    timezone,
)

from app.graph_engine.features import (
    StructuralFeatureExtractor,
)
from app.models.schemas import (
    CommentNode,
    ThreadData,
)
from app.services.thread_graph import (
    ThreadGraph,
)


def make_thread() -> ThreadData:
    start = datetime(
        2026,
        11,
        2,
        16,
        0,
        tzinfo=timezone.utc,
    )

    nodes = [
        CommentNode(
            id="root",
            author="alice",
            text="Root comment",
            timestamp=start,
            parent_id=None,
        ),
        CommentNode(
            id="reply_a",
            author="bob",
            text="First reply",
            timestamp=start + timedelta(
                seconds=10
            ),
            parent_id="root",
        ),
        CommentNode(
            id="reply_b",
            author="charlie",
            text="Second reply",
            timestamp=start + timedelta(
                seconds=20
            ),
            parent_id="root",
        ),
        CommentNode(
            id="deep_reply",
            author="bob",
            text="Deep reply",
            timestamp=start + timedelta(
                seconds=25
            ),
            parent_id="reply_a",
        ),
    ]

    return ThreadData(
        thread_id="feature_test",
        platform="reddit",
        title="Feature test",
        nodes=nodes,
    )


def test_structural_features():
    thread = make_thread()

    graph = ThreadGraph(
        thread
    )

    features = (
        StructuralFeatureExtractor(
            graph
        ).extract()
    )

    assert features["root"][
        "subtree_size"
    ] == 4

    assert features["reply_a"][
        "subtree_size"
    ] == 2

    assert features["reply_b"][
        "subtree_size"
    ] == 1

    assert features["reply_b"][
        "sibling_count"
    ] == 1

    assert features["reply_b"][
        "is_cross_reply"
    ] is True

    assert features["deep_reply"][
        "branch_root_id"
    ] == "reply_a"

    assert features["reply_b"][
        "branch_root_id"
    ] == "reply_b"

    assert features["root"][
        "branch_root_id"
    ] == "root"

    assert features["reply_a"][
        "chronological_index"
    ] == 1

    assert features["deep_reply"][
        "chronological_index"
    ] == 3

    assert features["reply_a"][
        "reply_velocity"
    ] == 10.0

    assert 0.0 <= features[
        "reply_a"
    ]["structural_impact_score"] <= 1.0