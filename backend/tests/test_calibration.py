from datetime import (
    datetime,
    timedelta,
    timezone,
)

from app.early_warning.engine import (
    EarlyWarningEngine,
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
                if toxicity >= 0.4
                else "neutral"
            ),

            scores={
                "anger": toxicity,

                "disgust": (
                    toxicity * 0.8
                ),

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

            character_count=32,

            avg_word_length=4.0,

            caps_ratio=0.0,

            exclamation_density=(
                0.08
                if toxicity >= 0.5
                else 0.0
            ),

            question_density=0.0,

            negation_density=(
                0.18
                if toxicity >= 0.3
                else 0.02
            ),

            hedging_score=(
                0.01
                if toxicity >= 0.3
                else 0.30
            ),

            pronoun_shift=(
                0.70
                if toxicity >= 0.3
                else 0.05
            ),

            lexical_diversity=(
                0.45
                if toxicity >= 0.3
                else 0.85
            ),

            profanity_score=(
                0.15
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
        0.76,
        0.74,
        0.72,
        0.70,
        0.68,
        0.65,
        0.62,
        0.58,
        0.54,
        0.20,
        -0.20,
        -0.48,
        -0.65,
        -0.75,
        -0.82,
        -0.86,
    ]

    toxicities = [
        0.01,
        0.01,
        0.01,
        0.02,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.09,
        0.16,
        0.28,
        0.42,
        0.55,
        0.73,
        0.79,
    ]

    nodes = []

    for index, (
        sentiment,
        toxicity,
    ) in enumerate(
        zip(
            sentiments,
            toxicities,
        )
    ):

        nodes.append(
            CommentNode(
                id=f"comment_{index + 1}",

                author=(
                    f"user_{index % 3}"
                ),

                text=(
                    f"Comment {index + 1}"
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
                    sentiment,
                    toxicity,
                ),

                features={
                    "reply_velocity_score": (
                        0.78
                        if index >= 9
                        else 0.12
                    ),

                    "drift_score": (
                        0.60
                        if index >= 10
                        else 0.08
                    ),

                    "change_point_probability": (
                        0.58
                        if index >= 10
                        else 0.05
                    ),
                },
            )
        )

    return ThreadData(
        thread_id="calibration_test",

        platform="reddit",

        title="Calibration test",

        nodes=nodes,
    )


def test_warning_is_inside_prediction_horizon():

    result = (
        EarlyWarningEngine().analyze(
            make_thread()
        )
    )

    warning = result.early_warning

    assert warning is not None

    assert (
        warning.realized_toxicity_index
        == 14
    )

    assert (
        warning.warning_was_early
        is True
    )

    assert (
        warning.earliest_warning_lead_comments
        is not None
    )

    assert (
        1
        <= warning
        .earliest_warning_lead_comments
        <= 3
    )

    assert (
        warning.earliest_warning_index
        is not None
    )

    assert (
        warning.earliest_warning_index
        >= 11
    )


def test_high_toxicity_means_high_current_risk():

    result = (
        EarlyWarningEngine().analyze(
            make_thread()
        )
    )

    warning = result.early_warning

    assert warning is not None

    assert (
        warning.current_risk
        >= 0.80
    )

    assert (
        warning.current_risk_level
        == "critical"
    )