from app.analyzers.emotion import EmotionAnalyzer
from app.analyzers.sentiment import SentimentAnalyzer
from app.analyzers.toxicity import ToxicityAnalyzer
from app.config import settings


def test_analyzers_have_local_fallback(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_MODE", "heuristic")

    texts = [
        "I agree, this is a great discussion.",
        "You are an idiot. Shut the fuck up!",
    ]

    sentiment = SentimentAnalyzer()
    toxicity = ToxicityAnalyzer()
    emotion = EmotionAnalyzer()

    sentiment_result = sentiment.analyze(texts)
    toxicity_result = toxicity.analyze(texts)
    emotion_result = emotion.analyze(texts)

    assert sentiment.mode == "heuristic"
    assert toxicity.mode == "heuristic"
    assert emotion.mode == "heuristic"

    assert sentiment_result[0]["sentiment_score"] > sentiment_result[1]["sentiment_score"]
    assert toxicity_result[1]["toxicity"] > toxicity_result[0]["toxicity"]
    assert emotion_result[1]["negative_emotion_intensity"] > emotion_result[0]["negative_emotion_intensity"]
