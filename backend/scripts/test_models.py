from app.analyzers.emotion import EmotionAnalyzer
from app.analyzers.sentiment import SentimentAnalyzer
from app.analyzers.toxicity import ToxicityAnalyzer


TEST_SENTENCES = [
    "I absolutely love this! This is amazing.",
    "I think this is okay.",
    "I completely disagree with you.",
    "You are an idiot and nobody wants to hear you.",
    "Shut up. You have no idea what you're talking about.",
    "Go away and leave me alone.",
    "Thank you for explaining that. I learned something.",
]


def main() -> None:
    print("Loading models...")
    print()

    sentiment = SentimentAnalyzer()
    toxicity = ToxicityAnalyzer()
    emotion = EmotionAnalyzer()

    print("=" * 80)
    print("CEREBRO MODEL SANITY TEST")
    print("=" * 80)

    for index, text in enumerate(TEST_SENTENCES, start=1):
        sentiment_result = sentiment.analyze([text])[0]
        toxicity_result = toxicity.analyze([text])[0]
        emotion_result = emotion.analyze([text])[0]

        print()
        print(f"TEST {index}")
        print("-" * 80)
        print(f"Text: {text}")
        print()

        print(
            f"Sentiment: "
            f"{sentiment_result['label']} "
            f"({sentiment_result['sentiment_score']:+.3f})"
        )

        print(
            f"Toxicity: "
            f"{toxicity_result['toxicity']:.3f}"
        )

        print(
            f"Insult: "
            f"{toxicity_result['insult']:.3f}"
        )

        print(
            f"Threat: "
            f"{toxicity_result['threat']:.3f}"
        )

        print(
            f"Obscene: "
            f"{toxicity_result['obscene']:.3f}"
        )

        print(
            f"Identity attack: "
            f"{toxicity_result['identity_attack']:.3f}"
        )

        print(
            f"Dominant emotion: "
            f"{emotion_result['dominant_emotion']}"
        )

        print("Emotion distribution:")

        for emotion_name, score in (
            emotion_result["scores"].items()
        ):
            print(
                f"  {emotion_name:10s}: {score:.3f}"
            )

    print()
    print("=" * 80)
    print("MODEL SANITY TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()