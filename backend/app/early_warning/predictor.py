from __future__ import annotations

from math import exp


class EarlyWarningPredictor:
    """
    Transparent precursor-risk model.

    The predictor only uses information available at the current comment.
    It does not use future comments to generate the forecast.
    """

    HORIZON = 3

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                value,
                1.0,
            ),
        )

    @staticmethod
    def _sigmoid(
        value: float,
    ) -> float:

        if value >= 0:

            z = exp(
                -value
            )

            return 1.0 / (
                1.0 + z
            )

        z = exp(value)

        return z / (
            1.0 + z
        )

    def predict(
        self,
        features: dict[str, float],
        warm: bool,
    ) -> dict:

        sentiment_slope = min(
            max(
                -features.get(
                    "sentiment_slope",
                    0.0,
                ),
                0.0,
            )
            / 0.20,
            1.0,
        )

        toxicity_slope = min(
            max(
                features.get(
                    "toxicity_slope",
                    0.0,
                ),
                0.0,
            )
            / 0.15,
            1.0,
        )

        emotion_slope = min(
            max(
                features.get(
                    "negative_emotion_slope",
                    0.0,
                ),
                0.0,
            )
            / 0.20,
            1.0,
        )

        reply_acceleration = min(
            max(
                features.get(
                    "reply_acceleration",
                    0.0,
                ),
                0.0,
            )
            / 0.25,
            1.0,
        )

        pronoun_shift = min(
            max(
                features.get(
                    "pronoun_shift_slope",
                    0.0,
                ),
                0.0,
            )
            / 0.20,
            1.0,
        )

        negation = min(
            max(
                features.get(
                    "negation_slope",
                    0.0,
                ),
                0.0,
            )
            / 0.15,
            1.0,
        )

        hedging = min(
            max(
                -features.get(
                    "hedging_slope",
                    0.0,
                ),
                0.0,
            )
            / 0.15,
            1.0,
        )

        diversity = min(
            max(
                -features.get(
                    "lexical_diversity_slope",
                    0.0,
                ),
                0.0,
            )
            / 0.15,
            1.0,
        )

        drift = min(
            max(
                features.get(
                    "current_drift_score",
                    0.0,
                ),
                0.0,
            ),
            1.0,
        )

        change_probability = min(
            max(
                features.get(
                    "current_change_probability",
                    0.0,
                ),
                0.0,
            ),
            1.0,
        )

        signal_count = int(
            features.get(
                "signal_count",
                0,
            )
        )

        current_toxicity = min(
            max(
                features.get(
                    "current_toxicity",
                    0.0,
                ),
                0.0,
            ),
            1.0,
        )

        raw_score = (
            1.90
            * sentiment_slope

            + 1.55
            * toxicity_slope

            + 1.05
            * emotion_slope

            + 0.70
            * reply_acceleration

            + 0.90
            * pronoun_shift

            + 0.65
            * negation

            + 0.75
            * hedging

            + 0.45
            * diversity

            + 0.30
            * drift

            + 0.15
            * change_probability
        )

        probability = self._sigmoid(
            raw_score - 3.55
        )

        if not warm:
            probability = min(
                probability,
                0.25,
            )

        # One weak precursor should not be allowed to produce an actionable
        # warning. We require multiple independent signals.
        if signal_count < 2:
            probability = min(
                probability,
                0.40,
            )

        elif signal_count < 3:
            probability = min(
                probability,
                0.58,
            )

        # Once a realized high-toxicity comment is present, the current
        # thread state should correctly reflect that the event is underway.
        if current_toxicity >= 0.70:
            probability = max(
                probability,
                0.92,
            )

        elif current_toxicity >= 0.50:
            probability = max(
                probability,
                0.80,
            )

        elif current_toxicity >= 0.30:
            probability = max(
                probability,
                0.63,
            )

        probability = self._clamp(
            probability
        )

        if probability >= 0.80:
            risk_level = "critical"

        elif probability >= 0.55:
            risk_level = "warning"

        elif probability >= 0.30:
            risk_level = "watch"

        else:
            risk_level = "safe"

        if risk_level == "critical":
            predicted_comments = 1

        elif risk_level == "warning":
            predicted_comments = 2

        elif risk_level == "watch":
            predicted_comments = 3

        else:
            predicted_comments = None

        return {
            "forecast_probability": round(
                probability,
                4,
            ),

            "risk_level": risk_level,

            "predicted_comments_to_event": (
                predicted_comments
            ),

            "model": (
                "explainable_precursor_risk"
            ),

            "horizon_comments": (
                self.HORIZON
            ),
        }