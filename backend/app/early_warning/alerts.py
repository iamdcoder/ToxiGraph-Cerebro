from __future__ import annotations


class EarlyWarningAlertBuilder:
    """
    Build operational early-warning summaries.

    A warning is considered useful for this system when it occurs inside
    the configured forecast horizon before a realized toxicity event.
    """

    def summarize(
        self,
        points: list[dict],
        evaluation: dict,
    ) -> dict:

        if not points:
            return {
                "current_risk": 0.0,
                "current_risk_level": "safe",
                "peak_forecast_probability": 0.0,
                "peak_forecast_comment_id": None,
                "earliest_warning_comment_id": None,
                "earliest_warning_index": None,
            }

        peak = max(
            points,
            key=lambda point: point[
                "forecast_probability"
            ],
        )

        actionable_warning_index = (
            evaluation.get(
                "earliest_warning_index"
            )
        )

        actionable_warning = None

        if (
            actionable_warning_index
            is not None
        ):

            actionable_warning = next(
                (
                    point
                    for point in points
                    if point[
                        "chronological_index"
                    ]
                    == actionable_warning_index
                ),
                None,
            )

        current = points[
            -1
        ]

        return {
            "current_risk": current[
                "forecast_probability"
            ],

            "current_risk_level": current[
                "risk_level"
            ],

            "peak_forecast_probability": peak[
                "forecast_probability"
            ],

            "peak_forecast_comment_id": peak[
                "comment_id"
            ],

            "earliest_warning_comment_id": (
                actionable_warning[
                    "comment_id"
                ]
                if actionable_warning
                else None
            ),

            "earliest_warning_index": (
                actionable_warning[
                    "chronological_index"
                ]
                if actionable_warning
                else None
            ),
        }

    @staticmethod
    def realized_lead_time(
        forecast_points: list[dict],
        nodes: list,
        toxicity_threshold: float = 0.70,
        warning_threshold: float = 0.55,
        horizon_comments: int = 3,
    ) -> dict:

        toxic_index = None

        for point, node in zip(
            forecast_points,
            nodes,
        ):

            toxicity = 0.0

            if node.analysis is not None:
                toxicity = float(
                    node
                    .analysis
                    .toxicity
                    .toxicity
                )

            if (
                toxicity
                >= toxicity_threshold
            ):

                toxic_index = point[
                    "chronological_index"
                ]

                break

        if toxic_index is None:

            return {
                "realized_toxicity_index": None,
                "earliest_warning_index": None,
                "earliest_warning_lead_comments": None,
                "warning_was_early": False,
            }

        earliest_allowed = max(
            0,
            toxic_index
            - horizon_comments,
        )

        latest_allowed = (
            toxic_index - 1
        )

        actionable_warnings = [
            point
            for point in forecast_points
            if (
                earliest_allowed
                <= point[
                    "chronological_index"
                ]
                <= latest_allowed
                and point[
                    "forecast_probability"
                ] >= warning_threshold
            )
        ]

        if not actionable_warnings:

            return {
                "realized_toxicity_index": toxic_index,
                "earliest_warning_index": None,
                "earliest_warning_lead_comments": None,
                "warning_was_early": False,
            }

        earliest_warning = min(
            actionable_warnings,
            key=lambda point: (
                point[
                    "chronological_index"
                ]
            ),
        )

        lead = (
            toxic_index
            - earliest_warning[
                "chronological_index"
            ]
        )

        return {
            "realized_toxicity_index": (
                toxic_index
            ),

            "earliest_warning_index": (
                earliest_warning[
                    "chronological_index"
                ]
            ),

            "earliest_warning_lead_comments": (
                lead
            ),

            "warning_was_early": (
                lead > 0
            ),
        }