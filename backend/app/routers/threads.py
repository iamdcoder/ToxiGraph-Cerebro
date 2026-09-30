from fastapi import APIRouter, HTTPException

from app.models.schemas import ThreadData, ThreadUploadResponse
from app.services.thread_graph import ThreadGraph


router = APIRouter()


@router.post("/validate", response_model=ThreadUploadResponse)
def validate_thread(thread: ThreadData) -> ThreadUploadResponse:
    try:
        graph = ThreadGraph(thread)
        normalized_thread = graph.to_thread_data()

        return ThreadUploadResponse(
            success=True,
            message="Thread validated and converted into a valid DAG.",
            thread=normalized_thread,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc