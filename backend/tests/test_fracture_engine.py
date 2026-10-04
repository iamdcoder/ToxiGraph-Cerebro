from datetime import (
    datetime,
    timedelta,
    timezone,
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
    anger: float = 0.0,
    disgust: float = 0.0,
) -> CommentAnalysis:

    negative_emotion = max(
        anger,
        disgust,
    )

    return CommentAnalysis(
        sentiment=SentimentAnalysis(
            label=(
                "positive"
                if sentiment > 0
                else "negative"
            ),
            score=abs(sentiment),
            sentiment_score=sentiment,
            negative=max(
                -sentiment,
                0,
            ),
            neutral=0.0,
            positive=max(
                sentiment,
                0,
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
            strongest_signal="toxicity",
            risk_level=(
                "high"
                if toxicity >= 0.5
                else "low"
            ),
        ),

        emotion=EmotionAnalysis(
            dominant_emotion=(
                "anger"
                if anger >= disgust
                and anger > 0
                else "neutral"
            ),

            scores={
                "anger": anger,
                "disgust": disgust,
                "fear": 0.0,
                "joy": 0.0,
                "neutral": max(
                    0.0,
                    1.0 - negative_emotion,
                ),
                "sadness": 0.0,
                "surprise": 0.0,
            },

            negative_emotion_intensity=(
                negative_emotion
            ),
        ),

        linguistic=LinguisticAnalysis(
            word_count=5,
            character_count=20,
            avg_word_length=4.0,
            caps_ratio=0.0,
            exclamation_density=0.0,
            question_density=0.0,
            negation_density=0.0,
            hedging_score=0.0,
            pronoun_shift=0.0,
            lexical_diversity=1.0,
            profanity_score=0.0,
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
        0.72,
        0.70,
        0.68,
        0.69,
        0.66,
        0.64,
        0.62,
        -0.15,
        -0.58,
        -0.72,
        -0.77,
        -0.80,
    ]

    nodes: list[CommentNode] = []

    for index, sentiment in enumerate(
        sentiments
    ):

        if index < 7:
            toxicity = 0.02
            anger = 0.05

        else:
            toxicity = min(
                0.05
                + (
                    index - 6
                )
                * 0.16,
                0.75,
            )

            anger = min(
                0.10
                + (
                    index - 6
                )
                * 0.15,
                0.75,
            )

        nodes.append(
            CommentNode(
                id=(
                    f"comment_{index + 1}"
                ),

                author=(
                    f"user_{index % 3}"
                ),

                text=(
                    f"Comment {index + 1}"
                ),

                timestamp=(
                    start
                    + timedelta(
                        seconds=index * 30
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
                    sentiment,
                    toxicity,
                    anger=anger,
                ),

                features={
                    "reply_velocity_score": (
                        0.20
                        if index < 7
                        else 0.80
                    ),
                },
            )
        )

    return ThreadData(
        thread_id="fracture_test",
        platform="reddit",
        title="Fracture test",
        nodes=nodes,
    )


def test_fracture_point_is_detected():

    result = DriftEngine().analyze(
        make_thread()
    )

    assert result.drift_analysis is not None

    timeline = (
        result
        .drift_analysis
        .global_timeline
    )

    assert timeline.fracture_point is not None

    fracture = (
        timeline.fracture_point
    )

    assert fracture.direction == (
        "negative"
    )

    assert fracture.confidence > 0

    assert fracture.severity > 0

    assert fracture.comment_id in {
        "comment_8",
        "comment_9",
        "comment_10",
        "comment_11",
    }

    assert len(
        fracture.signals
    ) > 0


def test_drift_signals_are_attached_to_nodes():

    result = DriftEngine().analyze(
        make_thread()
    )

    flagged_nodes = [
        node
        for node in result.nodes
        if node.features.get(
            "is_change_point"
        )
    ]

    assert flagged_nodes

    for node in flagged_nodes:

        assert node.features[
            "drift_score"
        ] > 0

        assert (
            "phase"
            in node.features
        )

        assert (
            "change_point_probability"
            in node.features
        )