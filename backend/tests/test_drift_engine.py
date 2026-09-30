from datetime import datetime, timedelta, timezone

from app.drift_engine.engine import DriftEngine
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
                if sentiment > 0
                else "negative"
            ),
            score=abs(sentiment),
            sentiment_score=sentiment,
            negative=max(-sentiment, 0),
            neutral=0.0,
            positive=max(sentiment, 0),
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
            risk_level="high"
            if toxicity >= 0.5
            else "low",
        ),
        emotion=EmotionAnalysis(
            dominant_emotion="neutral",
            scores={
                "anger": 0.0,
                "disgust": 0.0,
                "fear": 0.0,
                "joy": 0.0,
                "neutral": 1.0,
                "sadness": 0.0,
                "surprise": 0.0,
            },
            negative_emotion_intensity=0.0,
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


def test_drift_engine_detects_shift():
    start = datetime(
        2026,
        11,
        2,
        16,
        0,
        tzinfo=timezone.utc,
    )

    sentiments = [
        0.80,
        0.75,
        0.72,
        0.70,
        0.65,
        -0.40,
        -0.65,
        -0.75,
    ]

    nodes = []

    for index, sentiment in enumerate(
        sentiments
    ):
        parent_id = (
            None
            if index == 0
            else f"comment_{index}"
        )

        node = CommentNode(
            id=f"comment_{index + 1}",
            author=f"user_{index % 2}",
            text=f"Comment {index + 1}",
            timestamp=start + timedelta(
                seconds=index * 10
            ),
            parent_id=parent_id,
            analysis=make_analysis(
                sentiment,
                0.0
                if sentiment > 0
                else 0.05,
            ),
        )

        nodes.append(node)

    thread = ThreadData(
        thread_id="drift_test",
        platform="reddit",
        title="Drift test",
        nodes=nodes,
    )

    engine = DriftEngine()

    result = engine.analyze(thread)

    assert result.drift_analysis is not None

    timeline = (
        result
        .drift_analysis
        .global_timeline
        .timeline
    )

    assert len(timeline) == len(sentiments)

    assert (
        timeline[-1].sentiment_score
        < timeline[0].sentiment_score
    )

    assert (
        max(
            point.change_point_probability
            for point in timeline
        )
        > 0
    )

    assert (
        result
        .drift_analysis
        .strongest_change_point_id
        is not None
    )