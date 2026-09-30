import json
from pathlib import Path

from app.analyzers.pipeline import AnalysisPipeline
from app.models.schemas import ThreadData


def main() -> None:
    demo_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "demo"
        / "demo_thread.json"
    )

    with open(
        demo_path,
        "r",
        encoding="utf-8",
    ) as file:
        raw_data = json.load(file)

    thread = ThreadData.model_validate(
        raw_data
    )

    pipeline = AnalysisPipeline()

    analyzed = pipeline.analyze_thread(
        thread
    )

    print()
    print("=" * 60)
    print("CEREBRO PHASE 2 DEMO")
    print("=" * 60)

    for node in analyzed.nodes:
        analysis = node.analysis

        if analysis is None:
            continue

        print()
        print(
            f"[{node.id}] "
            f"{node.author}"
        )

        print(
            f"Text: {node.text}"
        )

        print(
            f"Sentiment: "
            f"{analysis.sentiment.label} "
            f"({analysis.sentiment.sentiment_score:+.3f})"
        )

        print(
            f"Toxicity: "
            f"{analysis.toxicity.toxicity:.3f} "
            f"[{analysis.toxicity.risk_level}]"
        )

        print(
            f"Emotion: "
            f"{analysis.emotion.dominant_emotion}"
        )

        print(
            f"Pronoun shift: "
            f"{analysis.linguistic.pronoun_shift:.3f}"
        )

        print(
            f"Negation density: "
            f"{analysis.linguistic.negation_density:.3f}"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()