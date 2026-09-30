from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


PlatformType = Literal[
    "reddit",
    "youtube",
    "x",
    "discord",
    "slack",
    "custom",
]


class SentimentAnalysis(BaseModel):
    label: str
    score: float
    sentiment_score: float
    negative: float
    neutral: float
    positive: float


class ToxicityAnalysis(BaseModel):
    toxicity: float
    severe_toxicity: float
    obscene: float
    identity_attack: float
    insult: float
    threat: float
    sexual_explicit: float
    strongest_signal: str
    risk_level: str


class EmotionAnalysis(BaseModel):
    dominant_emotion: str
    scores: dict[str, float]
    negative_emotion_intensity: float


class LinguisticAnalysis(BaseModel):
    word_count: int
    character_count: int
    avg_word_length: float

    caps_ratio: float
    exclamation_density: float
    question_density: float

    negation_density: float
    hedging_score: float
    pronoun_shift: float

    lexical_diversity: float
    profanity_score: float
    emoji_sentiment: float

    reply_length_ratio: float


class CommentAnalysis(BaseModel):
    sentiment: SentimentAnalysis
    toxicity: ToxicityAnalysis
    emotion: EmotionAnalysis
    linguistic: LinguisticAnalysis


class DriftPoint(BaseModel):
    comment_id: str
    chronological_index: int

    sentiment_score: float
    toxicity_score: float

    sentiment_delta: float
    rolling_sentiment: float
    toxicity_delta: float

    recent_change_probability: float
    run_length_zero_probability: float
    change_point_probability: float

    drift_score: float

    is_change_point: bool
    phase: str


class FracturePoint(BaseModel):
    comment_id: str
    chronological_index: int

    confidence: float
    severity: float

    change_point_probability: float
    drift_score: float

    sentiment_before: float
    sentiment_at: float
    sentiment_after: float

    toxicity_at: float

    phase_before: str
    phase_at: str
    phase_after: str

    direction: str

    persistence_score: float

    signals: list[str] = Field(
        default_factory=list
    )


class ThreadDriftTimeline(BaseModel):
    timeline: list[DriftPoint]

    change_points: list[DriftPoint]

    strongest_change_point_id: str | None = None

    max_drift_score: float

    current_phase: str

    fracture_point: FracturePoint | None = None


class BranchDrift(BaseModel):
    branch_root_id: str
    leaf_comment_id: str
    path_comment_ids: list[str]

    max_change_point_probability: float

    strongest_comment_id: str | None = None


class ThreadDriftAnalysis(BaseModel):
    global_timeline: ThreadDriftTimeline

    branches: list[BranchDrift]

    overall_change_point_count: int

    strongest_change_point_id: str | None = None


class AttributionCandidate(BaseModel):
    comment_id: str

    rank: int

    contribution_score: float

    baseline_drift_score: float
    counterfactual_drift_score: float

    drift_reduction: float

    baseline_change_probability: float
    counterfactual_change_probability: float

    sentiment_score: float
    toxicity_score: float

    evidence: list[str] = Field(
        default_factory=list
    )


class CausalChainLink(BaseModel):
    from_comment_id: str
    to_comment_id: str

    relationship: str

    influence_score: float

    evidence: list[str] = Field(
        default_factory=list
    )


class ThreadAttribution(BaseModel):
    turning_point_id: str | None = None

    primary_trigger_id: str | None = None

    confidence: str

    method: str = (
        "counterfactual_contribution"
    )

    summary: str = ""

    candidates: list[
        AttributionCandidate
    ] = Field(
        default_factory=list
    )

    causal_chain: list[
        CausalChainLink
    ] = Field(
        default_factory=list
    )


class EarlyWarningPoint(BaseModel):
    comment_id: str
    chronological_index: int

    forecast_probability: float

    risk_level: str

    predicted_comments_to_event: int | None = None

    horizon_comments: int

    precursor_signals: list[str] = Field(
        default_factory=list
    )

    precursor_features: dict[str, float] = Field(
        default_factory=dict
    )

    realized_toxicity_event: bool = False


class EarlyWarningAnalysis(BaseModel):
    timeline: list[
        EarlyWarningPoint
    ] = Field(
        default_factory=list
    )

    current_risk: float = 0.0

    current_risk_level: str = "safe"

    peak_forecast_probability: float = 0.0

    peak_forecast_comment_id: str | None = None

    earliest_warning_comment_id: str | None = None

    earliest_warning_index: int | None = None

    horizon_comments: int = 3

    realized_toxicity_index: int | None = None

    earliest_warning_lead_comments: int | None = None

    warning_was_early: bool = False


class CommentNode(BaseModel):
    id: str = Field(
        min_length=1
    )

    author: str = Field(
        min_length=1
    )

    text: str = Field(
        min_length=1
    )

    timestamp: datetime

    parent_id: str | None = None

    depth: int = Field(
        default=0,
        ge=0,
    )

    score: int = 0

    is_op: bool = False

    children_count: int = Field(
        default=0,
        ge=0,
    )

    features: dict[
        str,
        Any
    ] = Field(
        default_factory=dict
    )

    analysis: (
        CommentAnalysis | None
    ) = None


class ThreadEdge(BaseModel):
    source: str = Field(
        min_length=1
    )

    target: str = Field(
        min_length=1
    )

    edge_type: str = "reply_to"

    time_delta_seconds: float = Field(
        default=0.0,
        ge=0.0
    )


class ThreadData(BaseModel):
    thread_id: str = Field(
        min_length=1
    )

    platform: PlatformType

    title: str = ""

    created_at: datetime | None = None

    nodes: list[CommentNode]

    edges: list[ThreadEdge] = Field(
        default_factory=list
    )

    drift_analysis: (
        ThreadDriftAnalysis | None
    ) = None

    attribution: (
        ThreadAttribution | None
    ) = None

    early_warning: (
        EarlyWarningAnalysis | None
    ) = None

    @field_validator("nodes")
    @classmethod
    def nodes_must_not_be_empty(
        cls,
        value: list[CommentNode],
    ) -> list[CommentNode]:

        if not value:
            raise ValueError(
                "Thread must contain at least "
                "one comment."
            )

        ids = [
            node.id
            for node in value
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Comment IDs must be unique."
            )

        return value


class ThreadUploadResponse(BaseModel):
    success: bool
    message: str
    thread: ThreadData


class AnalysisResponse(BaseModel):
    success: bool
    message: str

    thread_id: str
    platform: PlatformType

    comment_count: int

    analyzed_thread: ThreadData


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str