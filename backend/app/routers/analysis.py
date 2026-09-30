from functools import lru_cache

from fastapi import (
    APIRouter,
    HTTPException,
)

from app.analyzers.pipeline import (
    AnalysisPipeline,
)

from app.attribution.engine import (
    AttributionEngine,
)

from app.drift_engine.engine import (
    DriftEngine,
)

from app.early_warning.engine import (
    EarlyWarningEngine,
)

from app.models.schemas import (
    AnalysisResponse,
    ThreadData,
)


router = APIRouter()


@lru_cache(maxsize=1)
def get_pipeline() -> AnalysisPipeline:
    return AnalysisPipeline()


@lru_cache(maxsize=1)
def get_drift_engine() -> DriftEngine:
    return DriftEngine()


@lru_cache(maxsize=1)
def get_attribution_engine() -> AttributionEngine:
    return AttributionEngine()


@lru_cache(maxsize=1)
def get_early_warning_engine() -> EarlyWarningEngine:
    return EarlyWarningEngine()


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
)
def analyze_thread(
    thread: ThreadData,
) -> AnalysisResponse:

    try:

        pipeline = get_pipeline()

        analyzed_thread = (
            pipeline.analyze_thread(
                thread
            )
        )

        drift_engine = (
            get_drift_engine()
        )

        analyzed_thread = (
            drift_engine.analyze(
                analyzed_thread
            )
        )

        attribution_engine = (
            get_attribution_engine()
        )

        analyzed_thread = (
            attribution_engine.analyze(
                analyzed_thread
            )
        )

        early_warning_engine = (
            get_early_warning_engine()
        )

        analyzed_thread = (
            early_warning_engine.analyze(
                analyzed_thread
            )
        )

        return AnalysisResponse(
            success=True,

            message=(
                "Thread analyzed using "
                "sentiment, toxicity, emotion, "
                "linguistic signals, graph-aware "
                "Bayesian drift detection, "
                "counterfactual trigger attribution, "
                "and explainable early-warning "
                "forecasting."
            ),

            thread_id=(
                analyzed_thread.thread_id
            ),

            platform=(
                analyzed_thread.platform
            ),

            comment_count=len(
                analyzed_thread.nodes
            ),

            analyzed_thread=(
                analyzed_thread
            ),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Analysis failed: {exc}"
            ),
        ) from exc