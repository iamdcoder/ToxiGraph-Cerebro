from __future__ import annotations

from app.early_warning.alerts import (
    EarlyWarningAlertBuilder,
)
from app.early_warning.precursors import (
    PrecursorSignalExtractor,
)
from app.early_warning.predictor import (
    EarlyWarningPredictor,
)
from app.models.schemas import (
    EarlyWarningAnalysis,
    EarlyWarningPoint,
    ThreadData,
)


class EarlyWarningEngine:
    """Generate a rolling forecast of near-future toxicity."""

    def __init__(self) -> None:

        self.precursors = (
            PrecursorSignalExtractor(
                window=5
            )
        )

        self.predictor = (
            EarlyWarningPredictor()
        )

        self.alerts = (
            EarlyWarningAlertBuilder()
        )

    def analyze(
        self,
        thread: ThreadData,
    ) -> ThreadData:

        nodes = sorted(
            thread.nodes,
            key=lambda node: (
                node.timestamp,
                node.id,
            ),
        )

        precursor_results = (
            self.precursors.extract(
                nodes
            )
        )

        points: list[
            EarlyWarningPoint
        ] = []

        plain_points: list[dict] = []

        for result, node in zip(
            precursor_results,
            nodes,
        ):

            features = dict(
                result["features"]
            )

            current_toxicity = 0.0

            if node.analysis is not None:
                current_toxicity = float(
                    node
                    .analysis
                    .toxicity
                    .toxicity
                )

            features[
                "current_toxicity"
            ] = current_toxicity

            prediction = (
                self.predictor.predict(
                    features,
                    warm=(
                        result[
                            "chronological_index"
                        ]
                        >= 4
                    ),
                )
            )

            point_dict = {
                **result,

                **prediction,

                "current_toxicity": round(
                    current_toxicity,
                    4,
                ),
            }

            plain_points.append(
                point_dict
            )

            points.append(
                EarlyWarningPoint(
                    comment_id=result[
                        "comment_id"
                    ],

                    chronological_index=result[
                        "chronological_index"
                    ],

                    forecast_probability=prediction[
                        "forecast_probability"
                    ],

                    risk_level=prediction[
                        "risk_level"
                    ],

                    predicted_comments_to_event=prediction[
                        "predicted_comments_to_event"
                    ],

                    horizon_comments=prediction[
                        "horizon_comments"
                    ],

                    precursor_signals=result[
                        "signals"
                    ],

                    precursor_features=result[
                        "features"
                    ],

                    realized_toxicity_event=(
                        current_toxicity
                        >= 0.70
                    ),
                )
            )

            node.features.update(
                {
                    "forecast_probability": prediction[
                        "forecast_probability"
                    ],

                    "risk_level": prediction[
                        "risk_level"
                    ],

                    "predicted_comments_to_event": prediction[
                        "predicted_comments_to_event"
                    ],

                    "precursor_signal_count": len(
                        result["signals"]
                    ),

                    "precursor_signals": result[
                        "signals"
                    ],
                }
            )

        evaluation = (
            self.alerts.realized_lead_time(
                plain_points,
                nodes,
                horizon_comments=(
                    self.predictor.HORIZON
                ),
            )
        )

        summary = (
            self.alerts.summarize(
                plain_points,
                evaluation,
            )
        )

        current = (
            points[-1]
            if points
            else None
        )

        maximum = (
            max(
                points,
                key=lambda point: (
                    point
                    .forecast_probability
                ),
            )
            if points
            else None
        )

        thread.early_warning = (
            EarlyWarningAnalysis(
                timeline=points,

                current_risk=(
                    current.forecast_probability
                    if current is not None
                    else 0.0
                ),

                current_risk_level=(
                    current.risk_level
                    if current is not None
                    else "safe"
                ),

                peak_forecast_probability=(
                    maximum.forecast_probability
                    if maximum is not None
                    else 0.0
                ),

                peak_forecast_comment_id=(
                    maximum.comment_id
                    if maximum is not None
                    else None
                ),

                earliest_warning_comment_id=summary[
                    "earliest_warning_comment_id"
                ],

                earliest_warning_index=summary[
                    "earliest_warning_index"
                ],

                horizon_comments=(
                    self.predictor.HORIZON
                ),

                realized_toxicity_index=evaluation[
                    "realized_toxicity_index"
                ],

                earliest_warning_lead_comments=evaluation[
                    "earliest_warning_lead_comments"
                ],

                warning_was_early=evaluation[
                    "warning_was_early"
                ],
            )
        )

        return thread