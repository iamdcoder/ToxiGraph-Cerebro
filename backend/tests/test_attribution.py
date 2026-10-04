from datetime import (
    datetime,
    timedelta,
    timezone,
)

from app.analyzers.linguistic import (
    LinguisticAnalyzer,
)
from app.attribution.engine import (
    AttributionEngine,
)
from app.drift_engine.engine import (
    DriftEngine,
)
from app.models.schemas import (
    CommentAnalysis,
    CommentNode,
    EmotionAnalysis,
    LinguisticAnalysis,
    SentimentAnalysis,
    ThreadData,
    ToxicityAnalysis,
)


def make_analysis(
    sentiment: float,
    toxicity: float,
    pronoun_shift: float = 0.0,
    negation_density: float = 0.0,
) -> CommentAnalysis:

    return CommentAnalysis(
        sentiment=SentimentAnalysis(
            label=(
                "positive"
                if sentiment >= 0
                else "negative"
            ),

            score=abs(
                sentiment
            ),

            sentiment_score=sentiment,

            negative=max(
                -sentiment,
                0.0,
            ),

            neutral=0.0,

            positive=max(
                sentiment,
                0.0,
            ),
        ),

        toxicity=ToxicityAnalysis(
            toxicity=toxicity,

            severe_toxicity=toxicity,

            obscene=toxicity,

            identity_attack=0.0,

            insult=toxicity,

            threat=0.0,

            sexual_explicit=0.0,

            strongest_signal=(
                "toxicity"
            ),

            risk_level=(
                "high"
                if toxicity >= 0.5
                else "low"
            ),
        ),

        emotion=EmotionAnalysis(
            dominant_emotion="neutral",

            scores={
                "anger": toxicity,
                "disgust": toxicity * 0.7,
                "fear": 0.0,
                "joy": max(
                    sentiment,
                    0.0,
                ),
                "neutral": max(
                    0.0,
                    1.0 - toxicity,
                ),
                "sadness": 0.0,
                "surprise": 0.0,
            },

            negative_emotion_intensity=(
                toxicity
            ),
        ),

        linguistic=LinguisticAnalysis(
            word_count=8,

            character_count=30,

            avg_word_length=4.0,

            caps_ratio=0.0,

            exclamation_density=(
                0.10
                if toxicity >= 0.5
                else 0.0
            ),

            question_density=0.0,

            negation_density=(
                negation_density
            ),

            hedging_score=(
                0.0
                if toxicity >= 0.5
                else 0.3
            ),

            pronoun_shift=(
                pronoun_shift
            ),

            lexical_diversity=0.75,

            profanity_score=(
                0.20
                if toxicity >= 0.6
                else 0.0
            ),

            emoji_sentiment=0.0,

            reply_length_ratio=1.0,
        ),
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

    sentiments = [
        0.75,
        0.72,
        0.70,
        0.67,
        0.64,
        0.61,
        0.58,
        0.53,
        -0.20,
        -0.65,
        -0.75,
        -0.80,
    ]

    toxicity = [
        0.01,
        0.01,
        0.02,
        0.02,
        0.02,
        0.03,
        0.03,
        0.04,
        0.18,
        0.55,
        0.68,
        0.72,
    ]

    nodes: list[
        CommentNode
    ] = []

    for index, (
        sentiment,
        toxicity_score,
    ) in enumerate(
        zip(
            sentiments,
            toxicity,
        )
    ):

        is_trigger = (
            index == 7
        )

        node = CommentNode(
            id=(
                f"comment_{index + 1}"
            ),

            author=(
                f"user_{index % 2}"
            ),

            text=(
                "Personal accusation."
                if is_trigger
                else (
                    f"Comment {index + 1}"
                )
            ),

            timestamp=(
                start
                + timedelta(
                    seconds=index * 15
                )
            ),

            parent_id=(
                None
                if index == 0
                else (
                    f"comment_{index}"
                )
            ),

            analysis=make_analysis(
                sentiment=sentiment,
                toxicity=toxicity_score,
                pronoun_shift=(
                    0.85
                    if is_trigger
                    else 0.0
                ),
                negation_density=(
                    0.25
                    if is_trigger
                    else 0.0
                ),
            ),

            features={
                "reply_velocity_score": (
                    0.85
                    if index >= 7
                    else 0.20
                ),

                "sentiment_delta": (
                    sentiments[index]
                    - sentiments[index - 1]
                    if index > 0
                    else 0.0
                ),

                "toxicity_delta": (
                    toxicity[index]
                    - toxicity[index - 1]
                    if index > 0
                    else 0.0
                ),
            },
        )

        nodes.append(node)

    return ThreadData(
        thread_id="attribution_test",
        platform="reddit",
        title="Attribution test",
        nodes=nodes,
    )


def test_counterfactual_attribution():

    thread = make_thread()

    thread = DriftEngine().analyze(
        thread
    )

    result = AttributionEngine().analyze(
        thread
    )

    assert result.attribution is not None

    attribution = (
        result.attribution
    )

    assert (
        attribution.primary_trigger_id
        is not None
    )

    assert (
        attribution.turning_point_id
        is not None
    )

    assert (
        attribution.method
        == "counterfactual_contribution"
    )

    assert attribution.candidates

    assert (
        attribution.candidates[0].rank
        == 1
    )

    assert (
        attribution.candidates[0]
        .contribution_score
        >= 0
    )


def test_trigger_fields_are_attached_to_nodes():

    thread = make_thread()

    thread = DriftEngine().analyze(
        thread
    )

    result = AttributionEngine().analyze(
        thread
    )

    trigger_nodes = [
        node
        for node in result.nodes
        if node.features.get(
            "is_trigger_candidate",
            False,
        )
    ]

    assert trigger_nodes

    primary_nodes = [
        node
        for node in result.nodes
        if node.features.get(
            "is_primary_trigger",
            False,
        )
    ]

    assert len(
        primary_nodes
    ) == 1

    primary = primary_nodes[0]

    assert (
        primary.features[
            "trigger_rank"
        ]
        == 1
    )

    assert (
        primary.features[
            "trigger_contribution_score"
        ]
        >= 0
    )