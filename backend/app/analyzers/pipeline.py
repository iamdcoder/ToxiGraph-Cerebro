from __future__ import annotations

from collections import Counter

from app.analyzers.emotion import (
    EmotionAnalyzer,
)
from app.analyzers.linguistic import (
    LinguisticAnalyzer,
)
from app.analyzers.sentiment import (
    SentimentAnalyzer,
)
from app.analyzers.toxicity import (
    ToxicityAnalyzer,
)
from app.graph_engine.features import (
    StructuralFeatureExtractor,
)
from app.config import settings
from app.models.schemas import (
    CommentAnalysis,
    EmotionAnalysis,
    LinguisticAnalysis,
    SentimentAnalysis,
    ThreadData,
    ToxicityAnalysis,
)
from app.services.thread_graph import (
    ThreadGraph,
)


class AnalysisPipeline:
    """
    Run NLP analysis and merge it with structural
    thread intelligence.
    """

    def __init__(self) -> None:
        print(
            "Loading sentiment model..."
        )
        self.sentiment = (
            SentimentAnalyzer()
        )

        print(
            "Loading toxicity model..."
        )
        self.toxicity = (
            ToxicityAnalyzer()
        )

        print(
            "Loading emotion model..."
        )
        self.emotion = (
            EmotionAnalyzer()
        )

        self.linguistic = (
            LinguisticAnalyzer()
        )

        print(
            "CEREBRO NLP + graph intelligence "
            "pipeline ready."
        )

    @property
    def metadata(self) -> dict[str, object]:
        modes = {
            "sentiment": self.sentiment.mode,
            "toxicity": self.toxicity.mode,
            "emotion": self.emotion.mode,
        }

        unique_modes = set(modes.values())
        if unique_modes == {"transformer"}:
            overall_mode = "transformer"
        elif unique_modes == {"heuristic"}:
            overall_mode = "heuristic"
        else:
            overall_mode = "hybrid"

        return {
            "requested_mode": settings.MODEL_MODE,
            "overall_mode": overall_mode,
            "sentiment_mode": modes["sentiment"],
            "toxicity_mode": modes["toxicity"],
            "emotion_mode": modes["emotion"],
            "fallback_active": "heuristic" in unique_modes,
        }

    def analyze_thread(
        self,
        thread: ThreadData,
    ) -> ThreadData:

        graph = ThreadGraph(
            thread
        )

        normalized_thread = (
            graph.to_thread_data()
        )

        nodes = (
            normalized_thread.nodes
        )

        texts = [
            node.text
            for node in nodes
        ]

        sentiment_results = (
            self.sentiment.analyze(
                texts
            )
        )

        toxicity_results = (
            self.toxicity.analyze(
                texts
            )
        )

        emotion_results = (
            self.emotion.analyze(
                texts
            )
        )

        structural_features = (
            StructuralFeatureExtractor(
                graph
            ).extract()
        )

        author_counts = Counter(
            node.author
            for node in nodes
        )

        parent_texts = {
            node.id: (
                graph.nodes[
                    node.parent_id
                ].text
                if node.parent_id is not None
                else None
            )
            for node in nodes
        }

        for index, node in enumerate(
            nodes
        ):
            linguistic_result = (
                self.linguistic.analyze(
                    text=node.text,
                    parent_text=(
                        parent_texts[
                            node.id
                        ]
                    ),
                )
            )

            sentiment_result = (
                sentiment_results[index]
            )

            toxicity_result = (
                toxicity_results[index]
            )

            emotion_result = (
                emotion_results[index]
            )

            node.analysis = (
                CommentAnalysis(
                    sentiment=(
                        SentimentAnalysis(
                            **sentiment_result
                        )
                    ),
                    toxicity=(
                        ToxicityAnalysis(
                            **toxicity_result
                        )
                    ),
                    emotion=(
                        EmotionAnalysis(
                            **emotion_result
                        )
                    ),
                    linguistic=(
                        LinguisticAnalysis(
                            **linguistic_result
                        )
                    ),
                )
            )

            # Structural features are generated separately so the NLP
            # analyzers remain independent from the graph engine.
            node.features = {
                **structural_features[
                    node.id
                ],

                # Semantic features
                "sentiment_score": (
                    sentiment_result[
                        "sentiment_score"
                    ]
                ),

                "toxicity_score": (
                    toxicity_result[
                        "toxicity"
                    ]
                ),

                "anger_score": (
                    emotion_result[
                        "scores"
                    ].get(
                        "anger",
                        0.0,
                    )
                ),

                "disgust_score": (
                    emotion_result[
                        "scores"
                    ].get(
                        "disgust",
                        0.0,
                    )
                ),

                "fear_score": (
                    emotion_result[
                        "scores"
                    ].get(
                        "fear",
                        0.0,
                    )
                ),

                "joy_score": (
                    emotion_result[
                        "scores"
                    ].get(
                        "joy",
                        0.0,
                    )
                ),

                "sadness_score": (
                    emotion_result[
                        "scores"
                    ].get(
                        "sadness",
                        0.0,
                    )
                ),

                # Linguistic features
                "word_count": (
                    linguistic_result[
                        "word_count"
                    ]
                ),

                "character_count": (
                    linguistic_result[
                        "character_count"
                    ]
                ),

                "avg_word_length": (
                    linguistic_result[
                        "avg_word_length"
                    ]
                ),

                "caps_ratio": (
                    linguistic_result[
                        "caps_ratio"
                    ]
                ),

                "exclamation_density": (
                    linguistic_result[
                        "exclamation_density"
                    ]
                ),

                "question_density": (
                    linguistic_result[
                        "question_density"
                    ]
                ),

                "negation_density": (
                    linguistic_result[
                        "negation_density"
                    ]
                ),

                "hedging_score": (
                    linguistic_result[
                        "hedging_score"
                    ]
                ),

                "pronoun_shift": (
                    linguistic_result[
                        "pronoun_shift"
                    ]
                ),

                "lexical_diversity": (
                    linguistic_result[
                        "lexical_diversity"
                    ]
                ),

                "profanity_score": (
                    linguistic_result[
                        "profanity_score"
                    ]
                ),

                "reply_length_ratio": (
                    linguistic_result[
                        "reply_length_ratio"
                    ]
                ),

                "emoji_sentiment": (
                    linguistic_result[
                        "emoji_sentiment"
                    ]
                ),

                # Existing compatibility field
                "author_reply_count": (
                    author_counts[
                        node.author
                    ]
                ),
            }

        return normalized_thread