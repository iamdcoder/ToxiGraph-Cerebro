from fastapi import APIRouter

from app.config import settings
from app.models.schemas import HealthResponse


router = APIRouter()


@router.get("/", response_model=HealthResponse)
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.APP_NAME,
        version=settings.APP_VERSION,
    )