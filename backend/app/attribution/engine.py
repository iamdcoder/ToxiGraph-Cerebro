from __future__ import annotations

from app.attribution.causal_chain import (
    CausalChainBuilder,
)
from app.attribution.counterfactual import (
    CounterfactualAttributor,
)
from app.models.schemas import (
    ThreadAttribution,
    ThreadData,
)


class AttributionEngine:
    """
    Combine counterfactual contribution analysis with a readable
    escalation-chain representation.
    """

    def __init__(self) -> None:

        self.counterfactual = (
            CounterfactualAttributor()
        )

        self.chain_builder = (
            CausalChainBuilder()
        )

    @staticmethod
    def _confidence(
        candidates,
    ) -> str:

        if not candidates:
            return "low"

        top = candidates[0]

        if len(candidates) == 1:
            if top.contribution_score >= 0.65:
                return "high"

            if top.contribution_score >= 0.35:
                return "medium"

            return "low"

        second = candidates[1]

        margin = (
            top.contribution_score
            - second.contribution_score
        )

        if (
            top.contribution_score >= 0.65
            and margin >= 0.15
        ):
            return "high"

        if (
            top.contribution_score >= 0.35
            and margin >= 0.07
        ):
            return "medium"

        return "low"

    @staticmethod
    def _summary(
        turning_point_id: str,
        primary_trigger_id: str,
        confidence: str,
    ) -> str:

        confidence_text = {
            "high": "high-confidence",
            "medium": "moderate-confidence",
            "low": "lower-confidence",
        }.get(
            confidence,
            "measured",
        )

        return (
            f"Counterfactual analysis identified "
            f"{primary_trigger_id} as the strongest "
            f"contributing trigger near fracture point "
            f"{turning_point_id} with "
            f"{confidence_text} attribution."
        )

    def analyze(
        self,
        thread: ThreadData,
    ) -> ThreadData:

        drift_analysis = (
            thread.drift_analysis
        )

        if drift_analysis is None:
            thread.attribution = None
            return thread

        fracture = (
            drift_analysis
            .global_timeline
            .fracture_point
        )

        if fracture is None:
            thread.attribution = None
            return thread

        candidates = (
            self.counterfactual
            .rank_candidates(
                thread
            )
        )

        if not candidates:
            thread.attribution = (
                ThreadAttribution(
                    turning_point_id=(
                        fracture.comment_id
                    ),
                    primary_trigger_id=None,
                    confidence="low",
                    method=(
                        "counterfactual_contribution"
                    ),
                    summary=(
                        "No sufficiently supported "
                        "contributing trigger was found "
                        "around the detected fracture."
                    ),
                    candidates=[],
                    causal_chain=[],
                )
            )

            return thread

        primary = candidates[0]

        confidence = (
            self._confidence(
                candidates
            )
        )

        causal_chain = (
            self.chain_builder.build(
                nodes=thread.nodes,
                primary=primary,
                fracture_id=(
                    fracture.comment_id
                ),
            )
        )

        thread.attribution = (
            ThreadAttribution(
                turning_point_id=(
                    fracture.comment_id
                ),

                primary_trigger_id=(
                    primary.comment_id
                ),

                confidence=confidence,

                method=(
                    "counterfactual_contribution"
                ),

                summary=self._summary(
                    fracture.comment_id,
                    primary.comment_id,
                    confidence,
                ),

                candidates=candidates,

                causal_chain=causal_chain,
            )
        )

        # Expose attribution directly on graph nodes so the frontend can
        # highlight the trigger without joining another response object.
        for node in thread.nodes:

            node.features[
                "is_trigger_candidate"
            ] = any(
                candidate.comment_id
                == node.id
                for candidate in candidates
            )

            node.features[
                "trigger_rank"
            ] = next(
                (
                    candidate.rank
                    for candidate in candidates
                    if candidate.comment_id
                    == node.id
                ),
                None,
            )

            node.features[
                "trigger_contribution_score"
            ] = next(
                (
                    candidate.contribution_score
                    for candidate in candidates
                    if candidate.comment_id
                    == node.id
                ),
                0.0,
            )

            node.features[
                "is_primary_trigger"
            ] = (
                node.id
                == primary.comment_id
            )

        return thread