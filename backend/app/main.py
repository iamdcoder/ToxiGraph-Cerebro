from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import analysis, health, threads


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "CEREBRO — Sentiment Drift Intelligence. "
        "An explainable system for detecting emotional shifts "
        "and escalation patterns in social conversation threads."
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    health.router,
    prefix=f"{settings.API_PREFIX}/health",
    tags=["Health"],
)

app.include_router(
    threads.router,
    prefix=f"{settings.API_PREFIX}/threads",
    tags=["Threads"],
)

app.include_router(
    analysis.router,
    prefix=f"{settings.API_PREFIX}/analysis",
    tags=["Analysis"],
)


@app.get("/")
def root() -> dict:
    return {
        "name": "CEREBRO",
        "codename": "ToxiGraph",
        "version": settings.APP_VERSION,
        "message": "Sentiment Drift Intelligence API is running.",
    }