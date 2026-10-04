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
    pronoun_shift: float,
    negation: float,
    hedging: float,
    diversity: float,
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
                "disgust": toxicity * 0.8,
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

            exclamation_density=0.0,

            question_density=0.0,

            negation_density=negation,

            hedging_score=hedging,

            pronoun_shift=pronoun_shift,

            lexical_diversity=diversity,

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
        0.70,
        0.68,
        0.66,
        0.64,
        0.60,
        0.52,
        0.40,
        0.20,
        -0.10,
        -0.40,
        -0.68,
        -0.78,
    ]

    toxicities = [
        0.01,
        0.01,
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.12,
        0.20,
        0.42,
        0.72,
        0.78,
    ]

    nodes: list[
        CommentNode
    ] = []

    for index, (
        sentiment,
        toxicity,
    ) in enumerate(
        zip(
            sentiments,
            toxicities,
        )
    ):

        escalation = (
            index >= 5
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

                    toxicity=toxicity,

                    pronoun_shift=(
                        0.70
                        if escalation
                        else 0.10
                    ),

                    negation=(
                        0.22
                        if escalation
                        else 0.03
                    ),

                    hedging=(
                        0.01
                        if escalation
                        else 0.30
                    ),

                    diversity=(
                        0.45
                        if escalation
                        else 0.85
                    ),
                ),

                features={
                    "reply_velocity_score": (
                        0.80
                        if escalation
                        else 0.15
                    ),

                    "drift_score": (
                        0.55
                        if index >= 8
                        else 0.05
                    ),

                    "change_point_probability": (
                        0.60
                        if index >= 8
                        else 0.05
                    ),
                },
            )
        )

    return ThreadData(
        thread_id=(
            "early_warning_test"
        ),

        platform="reddit",

        title="Early warning test",

        nodes=nodes,
    )


def test_early_warning_produces_timeline():

    result = (
        EarlyWarningEngine().analyze(
            make_thread()
        )
    )

    assert (
        result.early_warning
        is not None
    )

    assert len(
        result
        .early_warning
        .timeline
    ) == 12

    assert (
        result
        .early_warning
        .peak_forecast_probability
        > 0
    )

    warning_points = [
        point
        for point in (
            result
            .early_warning
            .timeline
        )
        if point.risk_level
        in {
            "warning",
            "critical",
        }
    ]

    assert warning_points


def test_early_warning_can_report_realized_lead_time():

    result = (
        EarlyWarningEngine().analyze(
            make_thread()
        )
    )

    assert (
        result.early_warning
        is not None
    )

    assert (
        result
        .early_warning
        .realized_toxicity_index
        == 10
    )

    assert (
        result
        .early_warning
        .warning_was_early
        is True
    )

    assert (
        result
        .early_warning
        .earliest_warning_lead_comments
        is not None
    )

    assert (
        result
        .early_warning
        .earliest_warning_lead_comments
        > 0
    )


def test_features_are_attached_to_nodes():

    result = (
        EarlyWarningEngine().analyze(
            make_thread()
        )
    )

    for node in result.nodes:

        assert (
            "forecast_probability"
            in node.features
        )

        assert (
            "risk_level"
            in node.features
        )

        assert (
            "precursor_signal_count"
            in node.features
        )