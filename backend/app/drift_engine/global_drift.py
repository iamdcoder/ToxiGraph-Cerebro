from statistics import mean

from app.drift_engine.bocpd import (
    BayesianChangePointDetector,
)

from app.models.schemas import (
    CommentNode,
    DriftPoint,
    FracturePoint,
    ThreadDriftTimeline,
)


class GlobalDriftDetector:
    """
    Detect emotional regime changes across the whole conversation.

    BOCPD provides the probabilistic change-point signal.

    Fracture confirmation additionally evaluates whether the negative
    emotional movement persists after a candidate point. This reduces
    false positives from isolated negative comments.
    """

    def __init__(
        self,
        hazard_rate: float = 1 / 50,
        max_run_length: int = 100,
        warmup: int = 5,
        change_threshold: float = 0.38,
        minimum_peak_separation: int = 2,
    ) -> None:

        self.hazard_rate = hazard_rate

        self.max_run_length = (
            max_run_length
        )

        self.warmup = warmup

        self.change_threshold = (
            change_threshold
        )

        self.minimum_peak_separation = (
            minimum_peak_separation
        )

    @staticmethod
    def _clamp(
        value: float,
        minimum: float = 0.0,
        maximum: float = 1.0,
    ) -> float:

        return max(
            minimum,
            min(
                value,
                maximum,
            ),
        )

    @staticmethod
    def _mean(
        values: list[float],
    ) -> float:

        if not values:
            return 0.0

        return mean(values)

    @staticmethod
    def _phase(
        rolling_sentiment: float,
        toxicity: float,
    ) -> str:

        if toxicity >= 0.70:
            return "toxic"

        if (
            rolling_sentiment <= -0.25
            or toxicity >= 0.25
        ):
            return "contentious"

        if rolling_sentiment < 0:
            return "tension"

        return "constructive"

    def _drift_score(
        self,
        recent_change_probability: float,
        sentiment_drop: float,
        rolling_drop: float,
        toxicity_jump: float,
        negative_emotion_jump: float,
        reply_velocity_score: float,
    ) -> float:

        normalized_sentiment_drop = (
            self._clamp(
                sentiment_drop / 0.75
            )
        )

        normalized_rolling_drop = (
            self._clamp(
                rolling_drop / 0.50
            )
        )

        normalized_toxicity_jump = (
            self._clamp(
                toxicity_jump / 0.40
            )
        )

        normalized_emotion_jump = (
            self._clamp(
                negative_emotion_jump
                / 0.40
            )
        )

        velocity_signal = (
            self._clamp(
                reply_velocity_score
            )
        )

        score = (
            0.35
            * recent_change_probability

            + 0.24
            * normalized_sentiment_drop

            + 0.16
            * normalized_rolling_drop

            + 0.10
            * normalized_toxicity_jump

            + 0.10
            * normalized_emotion_jump

            + 0.05
            * velocity_signal
        )

        return self._clamp(
            score
        )

    def _fracture_signals(
        self,
        point: DriftPoint,
        persistence_score: float,
        toxicity_rise: float,
        negative_emotion_jump: float,
        reply_velocity_score: float,
    ) -> list[str]:

        signals: list[str] = []

        if (
            point
            .recent_change_probability
            >= 0.35
        ):
            signals.append(
                "Bayesian change-point evidence increased"
            )

        if (
            point
            .sentiment_delta
            <= -0.20
        ):
            signals.append(
                "Sharp negative sentiment movement"
            )

        if persistence_score >= 0.45:
            signals.append(
                "Sentiment deterioration persists after the transition"
            )

        if (
            point
            .sentiment_delta
            <= -0.30
        ):
            signals.append(
                "Large one-comment sentiment drop"
            )

        if toxicity_rise >= 0.12:
            signals.append(
                "Toxicity rises after the transition"
            )

        if negative_emotion_jump >= 0.15:
            signals.append(
                "Anger/disgust intensity increased"
            )

        if reply_velocity_score >= 0.65:
            signals.append(
                "Reply activity accelerated"
            )

        if not signals:
            signals.append(
                "Multiple drift signals converged"
            )

        return signals

    def _fracture_quality(
        self,
        index: int,
        timeline: list[DriftPoint],
        nodes: list[CommentNode],
    ) -> tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        float,
    ]:

        before_values = [
            point.sentiment_score
            for point in timeline[
                max(
                    0,
                    index - 4,
                ):index
            ]
        ]

        after_values = [
            point.sentiment_score
            for point in timeline[
                index + 1:
                index + 4
            ]
        ]

        if not after_values:

            after_values = [
                point.sentiment_score
                for point in timeline[
                    index + 1:
                ]
            ][:3]

        sentiment_before = self._mean(
            before_values
        )

        sentiment_after = self._mean(
            after_values
            or [
                timeline[index]
                .sentiment_score
            ]
        )

        sustained_drop = max(
            sentiment_before
            - sentiment_after,
            0.0,
        )

        previous_toxicity = [
            point.toxicity_score
            for point in timeline[
                max(
                    0,
                    index - 3,
                ):index
            ]
        ]

        next_toxicity = [
            point.toxicity_score
            for point in timeline[
                index + 1:
                index + 4
            ]
        ]

        toxicity_before = self._mean(
            previous_toxicity
            or [
                timeline[index]
                .toxicity_score
            ]
        )

        toxicity_after = self._mean(
            next_toxicity
            or [
                timeline[index]
                .toxicity_score
            ]
        )

        toxicity_rise = max(
            toxicity_after
            - toxicity_before,
            0.0,
        )

        emotion_before: list[
            float
        ] = []

        emotion_after: list[
            float
        ] = []

        for node in nodes[
            max(
                0,
                index - 3,
            ):index
        ]:

            if node.analysis is None:
                continue

            scores = (
                node
                .analysis
                .emotion
                .scores
            )

            emotion_before.append(
                (
                    scores.get(
                        "anger",
                        0.0,
                    )
                    + scores.get(
                        "disgust",
                        0.0,
                    )
                )
                / 2.0
            )

        for node in nodes[
            index + 1:
            index + 4
        ]:

            if node.analysis is None:
                continue

            scores = (
                node
                .analysis
                .emotion
                .scores
            )

            emotion_after.append(
                (
                    scores.get(
                        "anger",
                        0.0,
                    )
                    + scores.get(
                        "disgust",
                        0.0,
                    )
                )
                / 2.0
            )

        emotion_rise = max(
            self._mean(
                emotion_after
            )
            - self._mean(
                emotion_before
            ),
            0.0,
        )

        persistence_score = (
            self._clamp(
                sustained_drop / 0.60
            )
        )

        normalized_toxicity_rise = (
            self._clamp(
                toxicity_rise / 0.40
            )
        )

        normalized_emotion_rise = (
            self._clamp(
                emotion_rise / 0.35
            )
        )

        quality = (
            0.40
            * persistence_score

            + 0.25
            * timeline[index]
            .drift_score

            + 0.20
            * timeline[index]
            .change_point_probability

            + 0.10
            * normalized_toxicity_rise

            + 0.05
            * normalized_emotion_rise
        )

        return (
            self._clamp(
                quality
            ),
            sentiment_before,
            sentiment_after,
            toxicity_before,
            toxicity_after,
            persistence_score,
            emotion_rise,
        )

    def _select_change_points(
        self,
        candidates: list[DriftPoint],
    ) -> list[DriftPoint]:

        if not candidates:
            return []

        selected: list[
            DriftPoint
        ] = []

        for candidate in candidates:

            if not selected:

                selected.append(
                    candidate
                )

                continue

            previous = selected[-1]

            distance = (
                candidate
                .chronological_index
                - previous
                .chronological_index
            )

            if (
                distance
                <= self.minimum_peak_separation
            ):

                if (
                    candidate
                    .drift_score
                    > previous
                    .drift_score
                ):

                    selected[-1] = (
                        candidate
                    )

            else:

                selected.append(
                    candidate
                )

        return selected

    def _build_fracture_point(
        self,
        candidate: DriftPoint,
        timeline: list[DriftPoint],
        nodes: list[CommentNode],
    ) -> FracturePoint:

        index = (
            candidate
            .chronological_index
        )

        (
            quality,
            sentiment_before,
            sentiment_after,
            toxicity_before,
            toxicity_after,
            persistence_score,
            emotion_rise,
        ) = self._fracture_quality(
            index,
            timeline,
            nodes,
        )

        point_node = nodes[
            index
        ]

        phase_before = (
            timeline[
                index - 1
            ].phase
            if index > 0
            else candidate.phase
        )

        phase_after = (
            timeline[
                index + 1
            ].phase
            if index + 1
            < len(timeline)
            else candidate.phase
        )

        reply_velocity_score = float(
            point_node
            .features
            .get(
                "reply_velocity_score",
                0.0,
            )
        )

        toxicity_rise = max(
            toxicity_after
            - toxicity_before,
            0.0,
        )

        confidence = self._clamp(
            0.45
            * persistence_score

            + 0.25
            * candidate
            .drift_score

            + 0.15
            * candidate
            .change_point_probability

            + 0.10
            * self._clamp(
                toxicity_rise
                / 0.40
            )

            + 0.05
            * self._clamp(
                emotion_rise
                / 0.35
            )
        )

        direction_delta = (
            sentiment_after
            - sentiment_before
        )

        if direction_delta <= -0.10:
            direction = "negative"

        elif direction_delta >= 0.10:
            direction = "positive"

        else:
            direction = "mixed"

        signals = (
            self._fracture_signals(
                candidate,
                persistence_score,
                toxicity_rise,
                emotion_rise,
                reply_velocity_score,
            )
        )

        return FracturePoint(
            comment_id=(
                candidate.comment_id
            ),

            chronological_index=(
                index
            ),

            confidence=round(
                confidence,
                4,
            ),

            severity=round(
                quality,
                4,
            ),

            change_point_probability=round(
                candidate
                .change_point_probability,
                4,
            ),

            drift_score=round(
                candidate
                .drift_score,
                4,
            ),

            sentiment_before=round(
                sentiment_before,
                4,
            ),

            sentiment_at=round(
                candidate
                .sentiment_score,
                4,
            ),

            sentiment_after=round(
                sentiment_after,
                4,
            ),

            toxicity_at=round(
                candidate
                .toxicity_score,
                4,
            ),

            phase_before=(
                phase_before
            ),

            phase_at=(
                candidate.phase
            ),

            phase_after=(
                phase_after
            ),

            direction=(
                direction
            ),

            persistence_score=round(
                persistence_score,
                4,
            ),

            signals=signals,
        )

    def detect(
        self,
        nodes: list[CommentNode],
    ) -> ThreadDriftTimeline:

        ordered_nodes = sorted(
            nodes,
            key=lambda node: (
                node.timestamp,
                node.id,
            ),
        )

        detector = (
            BayesianChangePointDetector(
                hazard_rate=(
                    self.hazard_rate
                ),
                max_run_length=(
                    self.max_run_length
                ),
                recent_window=5,
            )
        )

        timeline: list[
            DriftPoint
        ] = []

        sentiment_history: list[
            float
        ] = []

        toxicity_history: list[
            float
        ] = []

        emotion_history: list[
            float
        ] = []

        for index, node in enumerate(
            ordered_nodes
        ):

            if node.analysis is None:
                raise ValueError(
                    "All comments must be analyzed "
                    "before drift detection."
                )

            sentiment = (
                node
                .analysis
                .sentiment
                .sentiment_score
            )

            toxicity = (
                node
                .analysis
                .toxicity
                .toxicity
            )

            emotions = (
                node
                .analysis
                .emotion
                .scores
            )

            negative_emotion = (
                emotions.get(
                    "anger",
                    0.0,
                )
                + emotions.get(
                    "disgust",
                    0.0,
                )
            ) / 2.0

            recent_change_probability = (
                detector.update(
                    sentiment
                )
            )

            run_length_zero_probability = (
                detector
                .last_run_length_zero_probability
            )

            previous_sentiment = (
                sentiment_history[-1]
                if sentiment_history
                else sentiment
            )

            sentiment_delta = (
                sentiment
                - previous_sentiment
            )

            rolling_sentiment = (
                self._mean(
                    sentiment_history[-4:]
                    + [sentiment]
                )
            )

            previous_rolling = (
                self._mean(
                    sentiment_history[-5:]
                )
                if sentiment_history
                else rolling_sentiment
            )

            rolling_drop = max(
                previous_rolling
                - rolling_sentiment,
                0.0,
            )

            sentiment_drop = max(
                previous_sentiment
                - sentiment,
                0.0,
            )

            previous_toxicity = (
                toxicity_history[-1]
                if toxicity_history
                else toxicity
            )

            toxicity_delta = (
                toxicity
                - previous_toxicity
            )

            previous_emotion = (
                emotion_history[-1]
                if emotion_history
                else negative_emotion
            )

            emotion_jump = max(
                negative_emotion
                - previous_emotion,
                0.0,
            )

            reply_velocity_score = float(
                node
                .features
                .get(
                    "reply_velocity_score",
                    0.0,
                )
            )

            drift_score = (
                self._drift_score(
                    recent_change_probability,
                    sentiment_drop,
                    rolling_drop,
                    max(
                        toxicity_delta,
                        0.0,
                    ),
                    emotion_jump,
                    reply_velocity_score,
                )
            )

            phase = self._phase(
                rolling_sentiment,
                toxicity,
            )

            warm = (
                index >= self.warmup
            )

            enough_signal = (
                recent_change_probability
                >= 0.22

                or sentiment_drop
                >= 0.25

                or toxicity_delta
                >= 0.12
            )

            is_change_point = (
                warm
                and enough_signal
                and drift_score
                >= self.change_threshold
            )

            timeline.append(
                DriftPoint(
                    comment_id=node.id,

                    chronological_index=index,

                    sentiment_score=round(
                        sentiment,
                        4,
                    ),

                    toxicity_score=round(
                        toxicity,
                        4,
                    ),

                    sentiment_delta=round(
                        sentiment_delta,
                        4,
                    ),

                    rolling_sentiment=round(
                        rolling_sentiment,
                        4,
                    ),

                    toxicity_delta=round(
                        toxicity_delta,
                        4,
                    ),

                    recent_change_probability=round(
                        recent_change_probability,
                        4,
                    ),

                    run_length_zero_probability=round(
                        run_length_zero_probability,
                        4,
                    ),

                    change_point_probability=round(
                        recent_change_probability,
                        4,
                    ),

                    drift_score=round(
                        drift_score,
                        4,
                    ),

                    is_change_point=(
                        is_change_point
                    ),

                    phase=phase,
                )
            )

            sentiment_history.append(
                sentiment
            )

            toxicity_history.append(
                toxicity
            )

            emotion_history.append(
                negative_emotion
            )

        candidates = [
            point
            for point in timeline
            if point.is_change_point
        ]

        change_points = (
            self._select_change_points(
                candidates
            )
        )

        selected_ids = {
            point.comment_id
            for point in change_points
        }

        for point in timeline:

            point.is_change_point = (
                point.comment_id
                in selected_ids
            )

        fracture_candidate = None
        best_quality = 0.0

        fracture_start = min(
            self.warmup,
            max(
                len(timeline) - 1,
                0,
            ),
        )

        for index in range(
            fracture_start,
            len(timeline),
        ):

            if (
                index
                >= len(timeline) - 1
            ):
                continue

            (
                quality,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = self._fracture_quality(
                index,
                timeline,
                ordered_nodes,
            )

            if quality > best_quality:

                best_quality = (
                    quality
                )

                fracture_candidate = (
                    timeline[index]
                )

        fracture_point = None

        if (
            fracture_candidate
            is not None
            and best_quality >= 0.30
        ):

            fracture_point = (
                self._build_fracture_point(
                    fracture_candidate,
                    timeline,
                    ordered_nodes,
                )
            )

        warm_timeline = timeline[
            self.warmup:
        ]

        strongest = None

        if warm_timeline:

            strongest = max(
                warm_timeline,
                key=lambda point: (
                    point.drift_score,
                    point.change_point_probability,
                ),
            )

        max_drift = (
            max(
                (
                    point.drift_score
                    for point in warm_timeline
                ),
                default=0.0,
            )
        )

        return ThreadDriftTimeline(
            timeline=timeline,

            change_points=(
                change_points
            ),

            strongest_change_point_id=(
                strongest.comment_id
                if strongest is not None
                else None
            ),

            max_drift_score=round(
                max_drift,
                4,
            ),

            current_phase=(
                timeline[-1].phase
                if timeline
                else "constructive"
            ),

            fracture_point=(
                fracture_point
            ),
        )