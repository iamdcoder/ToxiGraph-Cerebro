from datetime import datetime, timezone

from app.models.schemas import CommentNode, ThreadData
from app.services.thread_graph import ThreadGraph


def test_thread_graph_builds_correctly():
    thread = ThreadData(
        thread_id="test_001",
        platform="reddit",
        title="Test thread",
        nodes=[
            CommentNode(
                id="root",
                author="alice",
                text="Hello",
                timestamp=datetime(
                    2026, 11, 2, 16, 0, tzinfo=timezone.utc
                ),
                parent_id=None,
            ),
            CommentNode(
                id="reply_1",
                author="bob",
                text="Hi",
                timestamp=datetime(
                    2026, 11, 2, 16, 1, tzinfo=timezone.utc
                ),
                parent_id="root",
            ),
            CommentNode(
                id="reply_2",
                author="charlie",
                text="Hello both",
                timestamp=datetime(
                    2026, 11, 2, 16, 2, tzinfo=timezone.utc
                ),
                parent_id="reply_1",
            ),
        ],
    )

    graph = ThreadGraph(thread)
    result = graph.to_thread_data()

    assert len(result.nodes) == 3
    assert len(result.edges) == 2

    assert result.nodes[0].depth == 0
    assert result.nodes[1].depth == 1
    assert result.nodes[2].depth == 2