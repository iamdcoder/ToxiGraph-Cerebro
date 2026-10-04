import json
from pathlib import Path

from app.models.schemas import ThreadData
from app.services.thread_graph import ThreadGraph


ROOT = Path(__file__).parents[2]


def test_example_fixture_has_a_real_thread_shape():
    path = ROOT / "backend" / "data" / "examples" / "example_thread.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    thread = ThreadData.model_validate(payload)
    graph = ThreadGraph(thread)

    assert len(graph.nodes) >= 30
    assert len(graph.get_roots()) == 1
    assert len(graph.edges) == len(graph.nodes) - 1
    assert len(graph.children["comment_1"]) >= 3


def test_frontend_and_backend_example_fixtures_stay_in_sync():
    backend_path = ROOT / "backend" / "data" / "examples" / "example_thread.json"
    frontend_path = ROOT / "frontend" / "public" / "example_thread.json"

    backend_payload = json.loads(backend_path.read_text(encoding="utf-8"))
    frontend_payload = json.loads(frontend_path.read_text(encoding="utf-8"))

    assert frontend_payload == backend_payload
