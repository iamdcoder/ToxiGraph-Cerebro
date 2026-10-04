from __future__ import annotations

from statistics import mean
from typing import Any

from app.models.schemas import CommentNode


class PrecursorSignalExtractor:
    """
    Compute explainable changes in the conversation that may precede
    escalation into toxic behavior.
    """

    def __init__(
        self,
        window: int = 5,
    ) -> None:
        self.window = max(
            window,
            2,
        )

    @staticmethod
    def _mean(
        values: list[float],
    ) -> float:
        return (
            mean(values)
            if values
            else 0.0
        )

    @staticmethod
    def _slope(
        values: list[float],
    ) -> float:
        if len(values) < 2:
            return 0.0

        return (
            values[-1] - values[0]
        ) / max(
            len(values) - 1,
            1,
        )

    def extract(
        self,
        nodes: list[CommentNode],
    ) -> list[dict[str, Any]]:

        ordered = sorted(
            nodes,
            key=lambda node: (
                node.timestamp,
                node.id,
            ),
        )

        results: list[
            dict[str, Any]
        ] = []

        for index, node in enumerate(
            ordered
        ):
            current = ordered[
                max(
                    0,
                    index - self.window + 1,
                ):index + 1
            ]

            previous = ordered[
                max(
                    0,
                    index - self.window,
                ):index
            ]

            def sentiment_of(
                item: CommentNode,
            ) -> float:

                if item.analysis is not None:
                    return float(
                        item
                        .analysis
                        .sentiment
                        .sentiment_score
                    )

                return float(
                    item.features.get(
                        "sentiment_score",
                        0.0,
                    )
                )

            def toxicity_of(
                item: CommentNode,
            ) -> float:

                if item.analysis is not None:
                    return float(
                        item
                        .analysis
                        .toxicity
                        .toxicity
                    )

                return float(
                    item.features.get(
                        "toxicity_score",
                        0.0,
                    )
                )

            def anger_of(
                item: CommentNode,
            ) -> float:

                if item.analysis is not None:
                    return float(
                        item
                        .analysis
                        .emotion
                        .scores
                        .get(
                            "anger",
                            0.0,
                        )
                    )

                return float(
                    item.features.get(
                        "anger_score",
                        0.0,
                    )
                )

            def disgust_of(
                item: CommentNode,
            ) -> float:

                if item.analysis is not None:
                    return float(
                        item
                        .analysis
                        .emotion
                        .scores
                        .get(
                            "disgust",
                            0.0,
                        )
                    )

                return float(
                    item.features.get(
                        "disgust_score",
                        0.0,
                    )
                )

            def linguistic(
                item: CommentNode,
                key: str,
            ) -> float:

                if item.analysis is not None:
                    return float(
                        getattr(
                            item
                            .analysis
                            .linguistic,
                            key,
                        )
                    )

                return float(
                    item.features.get(
                        key,
                        0.0,
                    )
                )

            sentiments = [
                sentiment_of(item)
                for item in current
            ]

            previous_sentiments = [
                sentiment_of(item)
                for item in previous
            ]

            toxicities = [
                toxicity_of(item)
                for item in current
            ]

            previous_toxicities = [
                toxicity_of(item)
                for item in previous
            ]

            negative_emotions = [
                (
                    anger_of(item)
                    + disgust_of(item)
                ) / 2.0
                for item in current
            ]

            previous_negative_emotions = [
                (
                    anger_of(item)
                    + disgust_of(item)
                ) / 2.0
                for item in previous
            ]

            reply_speed = [
                float(
                    item.features.get(
                        "reply_velocity_score",
                        0.0,
                    )
                )
                for item in current
            ]

            previous_reply_speed = [
                float(
                    item.features.get(
                        "reply_velocity_score",
                        0.0,
                    )
                )
                for item in previous
            ]

            pronouns = [
                linguistic(
                    item,
                    "pronoun_shift",
                )
                for item in current
            ]

            previous_pronouns = [
                linguistic(
                    item,
                    "pronoun_shift",
                )
                for item in previous
            ]

            negations = [
                linguistic(
                    item,
                    "negation_density",
                )
                for item in current
            ]

            previous_negations = [
                linguistic(
                    item,
                    "negation_density",
                )
                for item in previous
            ]

            hedging = [
                linguistic(
                    item,
                    "hedging_score",
                )
                for item in current
            ]

            previous_hedging = [
                linguistic(
                    item,
                    "hedging_score",
                )
                for item in previous
            ]

            diversity = [
                linguistic(
                    item,
                    "lexical_diversity",
                )
                for item in current
            ]

            previous_diversity = [
                linguistic(
                    item,
                    "lexical_diversity",
                )
                for item in previous
            ]

            current_sentiment = (
                sentiments[-1]
            )

            current_toxicity = (
                toxicities[-1]
            )

            sentiment_slope = (
                self._slope(
                    sentiments
                )
            )

            toxicity_slope = (
                self._slope(
                    toxicities
                )
            )

            emotion_slope = (
                self._slope(
                    negative_emotions
                )
            )

            speed_slope = (
                self._slope(
                    reply_speed
                )
            )

            pronoun_slope = (
                self._slope(
                    pronouns
                )
            )

            negation_slope = (
                self._slope(
                    negations
                )
            )

            hedging_slope = (
                self._slope(
                    hedging
                )
            )

            diversity_slope = (
                self._slope(
                    diversity
                )
            )

            recent_sentiment_drop = (
                max(
                    self._mean(
                        previous_sentiments
                    )
                    - current_sentiment,
                    0.0,
                )
                if previous_sentiments
                else 0.0
            )

            recent_toxicity_rise = (
                max(
                    current_toxicity
                    - self._mean(
                        previous_toxicities
                    ),
                    0.0,
                )
                if previous_toxicities
                else 0.0
            )

            recent_emotion_rise = (
                max(
                    negative_emotions[-1]
                    - self._mean(
                        previous_negative_emotions
                    ),
                    0.0,
                )
                if previous_negative_emotions
                else 0.0
            )

            recent_speed_rise = (
                max(
                    reply_speed[-1]
                    - self._mean(
                        previous_reply_speed
                    ),
                    0.0,
                )
                if previous_reply_speed
                else 0.0
            )

            recent_pronoun_rise = (
                max(
                    pronouns[-1]
                    - self._mean(
                        previous_pronouns
                    ),
                    0.0,
                )
                if previous_pronouns
                else 0.0
            )

            recent_negation_rise = (
                max(
                    negations[-1]
                    - self._mean(
                        previous_negations
                    ),
                    0.0,
                )
                if previous_negations
                else 0.0
            )

            recent_hedging_drop = (
                max(
                    self._mean(
                        previous_hedging
                    )
                    - hedging[-1],
                    0.0,
                )
                if previous_hedging
                else 0.0
            )

            recent_diversity_drop = (
                max(
                    self._mean(
                        previous_diversity
                    )
                    - diversity[-1],
                    0.0,
                )
                if previous_diversity
                else 0.0
            )

            drift_score = float(
                node.features.get(
                    "drift_score",
                    0.0,
                )
            )

            change_probability = float(
                node.features.get(
                    "change_point_probability",
                    0.0,
                )
            )

            signals: list[str] = []

            if (
                sentiment_slope <= -0.08
                or recent_sentiment_drop >= 0.15
            ):
                signals.append(
                    "Sentiment is deteriorating"
                )

            if (
                toxicity_slope >= 0.05
                or recent_toxicity_rise >= 0.10
            ):
                signals.append(
                    "Toxicity momentum is rising"
                )

            if (
                speed_slope >= 0.08
                or recent_speed_rise >= 0.12
            ):
                signals.append(
                    "Replies are accelerating"
                )

            if (
                pronoun_slope >= 0.08
                or recent_pronoun_rise >= 0.12
            ):
                signals.append(
                    "Accusatory pronoun use is increasing"
                )

            if (
                negation_slope >= 0.05
                or recent_negation_rise >= 0.08
            ):
                signals.append(
                    "Negation and disagreement language is increasing"
                )

            if (
                hedging_slope <= -0.05
                or recent_hedging_drop >= 0.08
            ):
                signals.append(
                    "Hedging language is decreasing"
                )

            if (
                diversity_slope <= -0.05
                or recent_diversity_drop >= 0.08
            ):
                signals.append(
                    "Vocabulary diversity is narrowing"
                )

            if (
                emotion_slope >= 0.06
                or recent_emotion_rise >= 0.10
            ):
                signals.append(
                    "Anger/disgust intensity is increasing"
                )

            if (
                change_probability >= 0.35
                or drift_score >= 0.35
            ):
                signals.append(
                    "The thread is approaching a detected drift region"
                )

            precursor_features = {
                "sentiment_slope": round(
                    sentiment_slope,
                    4,
                ),

                "toxicity_slope": round(
                    toxicity_slope,
                    4,
                ),

                "negative_emotion_slope": round(
                    emotion_slope,
                    4,
                ),

                "reply_acceleration": round(
                    speed_slope,
                    4,
                ),

                "pronoun_shift_slope": round(
                    pronoun_slope,
                    4,
                ),

                "negation_slope": round(
                    negation_slope,
                    4,
                ),

                "hedging_slope": round(
                    hedging_slope,
                    4,
                ),

                "lexical_diversity_slope": round(
                    diversity_slope,
                    4,
                ),

                "recent_sentiment_drop": round(
                    recent_sentiment_drop,
                    4,
                ),

                "recent_toxicity_rise": round(
                    recent_toxicity_rise,
                    4,
                ),

                "recent_emotion_rise": round(
                    recent_emotion_rise,
                    4,
                ),

                "recent_reply_speed_rise": round(
                    recent_speed_rise,
                    4,
                ),

                "recent_pronoun_rise": round(
                    recent_pronoun_rise,
                    4,
                ),

                "recent_negation_rise": round(
                    recent_negation_rise,
                    4,
                ),

                "recent_hedging_drop": round(
                    recent_hedging_drop,
                    4,
                ),

                "recent_diversity_drop": round(
                    recent_diversity_drop,
                    4,
                ),

                "current_drift_score": round(
                    drift_score,
                    4,
                ),

                "current_change_probability": round(
                    change_probability,
                    4,
                ),

                "signal_count": len(
                    signals
                ),
            }

            results.append(
                {
                    "comment_id": node.id,
                    "chronological_index": index,
                    "signals": signals,
                    "features": precursor_features,
                }
            )

        return results