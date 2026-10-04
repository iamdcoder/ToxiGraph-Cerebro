from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/api/v1/health/")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["service"] == "CEREBRO"

def test_analysis_response_exposes_engine_metadata(monkeypatch):
    monkeypatch.setenv("MODEL_MODE", "heuristic")

    # The Settings object is initialized at import time, so this check uses
    # the already configured fallback mode in the test environment.
    from app.routers.analysis import get_pipeline

    pipeline = get_pipeline()
    metadata = pipeline.metadata

    assert metadata["overall_mode"] in {"transformer", "heuristic", "hybrid"}
    assert metadata["sentiment_mode"] in {"transformer", "heuristic"}
    assert metadata["toxicity_mode"] in {"transformer", "heuristic"}
    assert metadata["emotion_mode"] in {"transformer", "heuristic"}
