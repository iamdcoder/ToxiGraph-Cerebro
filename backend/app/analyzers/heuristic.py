from __future__ import annotations

import re
from math import exp
from typing import Any


POSITIVE_WORDS = {
    "agree",
    "agreed",
    "amazing",
    "benefit",
    "calm",
    "clear",
    "constructive",
    "enjoy",
    "fair",
    "good",
    "great",
    "helpful",
    "interesting",
    "love",
    "nice",
    "positive",
    "reasonable",
    "respect",
    "thanks",
    "thank",
    "useful",
    "welcome",
    "well",
}

NEGATIVE_WORDS = {
    "annoying",
    "attack",
    "attacking",
    "awful",
    "bad",
    "blame",
    "clueless",
    "crap",
    "disagree",
    "dumb",
    "fool",
    "hate",
    "idiot",
    "ignorant",
    "insulting",
    "insult",
    "ridiculous",
    "stupid",
    "terrible",
    "toxic",
    "unfair",
    "useless",
    "wrong",
    "accusation",
    "attacks",
    "away",
    "burnout",
    "expert",
    "hell",
    "hostile",
    "lecturing",
    "lecture",
    "personal",
    "pretending",
    "problem",
    "stop",
    "wasting",
}

TOXIC_WORDS = {
    "asshole",
    "bastard",
    "bullshit",
    "clueless",
    "damn",
    "dumb",
    "fuck",
    "fucking",
    "idiot",
    "moron",
    "ridiculous",
    "shit",
    "stupid",
}

THREAT_WORDS = {
    "destroy",
    "kill",
    "hurt",
    "threat",
    "threaten",
}

DISGUST_WORDS = {
    "disgust",
    "disgusting",
    "gross",
    "nasty",
    "nonsense",
    "revolting",
}

ANGER_WORDS = {
    "angry",
    "furious",
    "mad",
    "rage",
    "ridiculous",
    "stupid",
    "hate",
    "clueless",
    "idiot",
    "moron",
    "fuck",
    "fucking",
    "shut",
}

FEAR_WORDS = {
    "afraid",
    "fear",
    "scared",
    "worry",
    "worried",
}

JOY_WORDS = {
    "amazing",
    "enjoy",
    "fun",
    "good",
    "great",
    "love",
    "nice",
}

SADNESS_WORDS = {
    "bad",
    "depressed",
    "sad",
    "sorry",
    "upset",
}

TOKEN_RE = re.compile(r"[A-Za-z']+")


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def _ratio(hits: int, total: int) -> float:
    return hits / max(total, 1)


def _sigmoid(value: float) -> float:
    value = max(-20.0, min(20.0, value))
    return 1.0 / (1.0 + exp(-value))


def heuristic_sentiment(texts: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for text in texts:
        tokens = _tokens(text)
        positive_hits = sum(token in POSITIVE_WORDS for token in tokens)
        negative_hits = sum(token in NEGATIVE_WORDS for token in tokens)

        raw = (positive_hits - negative_hits) / max(len(tokens), 1)
        sentiment_score = max(-1.0, min(1.0, raw * 2.8))
        positive = _sigmoid(sentiment_score * 3.0) if sentiment_score > 0 else 0.15 + max(sentiment_score, -1.0) * -0.05
        negative = _sigmoid(-sentiment_score * 3.0) if sentiment_score < 0 else 0.15 + min(sentiment_score, 1.0) * -0.05
        neutral = max(0.0, 1.0 - min(0.92, positive + negative))

        if abs(sentiment_score) < 0.12:
            label = "neutral"
            positive = max(positive, 0.30)
            negative = max(negative, 0.30)
            neutral = max(neutral, 0.40)
        elif sentiment_score > 0:
            label = "positive"
        else:
            label = "negative"

        total = positive + neutral + negative
        output.append(
            {
                "label": label,
                "score": round(max(positive, negative, neutral) / max(total, 1e-9), 4),
                "sentiment_score": round(sentiment_score, 4),
                "negative": round(negative / max(total, 1e-9), 4),
                "neutral": round(neutral / max(total, 1e-9), 4),
                "positive": round(positive / max(total, 1e-9), 4),
            }
        )

    return output


def heuristic_toxicity(texts: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for text in texts:
        tokens = _tokens(text)
        total = max(len(tokens), 1)
        toxic_hits = sum(token in TOXIC_WORDS for token in tokens)
        insult_hits = sum(token in {"clueless", "dumb", "idiot", "moron", "stupid", "ridiculous"} for token in tokens)
        obscene_hits = sum(token in {"fuck", "fucking", "shit", "bullshit", "crap"} for token in tokens)
        threat_hits = sum(token in THREAT_WORDS for token in tokens)
        disgust_hits = sum(token in DISGUST_WORDS for token in tokens)

        intensity = (
            1.15 * toxic_hits
            + 0.9 * insult_hits
            + 0.75 * obscene_hits
            + 0.65 * threat_hits
            + 0.35 * disgust_hits
            + 0.35 * text.count("!")
            + 0.25 * text.count("?")
        ) / total

        toxicity = max(0.0, min(0.98, intensity * 1.9))

        # Strong profanity is a high-confidence toxicity cue in the
        # offline fallback. Keep the threshold aligned with the same
        # critical band used by the transformer path so local development can
        # exercise the full early-warning -> realized-event flow.
        if obscene_hits > 0:
            toxicity = max(toxicity, 0.74)
        elif insult_hits >= 2:
            toxicity = max(toxicity, 0.68)

        severe = min(0.98, toxicity * (1.15 if obscene_hits or threat_hits else 0.55))
        obscene = min(0.98, obscene_hits / total * 2.5)
        identity_attack = 0.0
        insult = min(0.98, insult_hits / total * 3.0)
        threat = min(0.98, threat_hits / total * 3.0)
        sexual_explicit = 0.0

        primary_scores = {
            "toxicity": toxicity,
            "severe_toxicity": severe,
            "obscene": obscene,
            "identity_attack": identity_attack,
            "insult": insult,
            "threat": threat,
            "sexual_explicit": sexual_explicit,
        }

        strongest_signal = max(primary_scores, key=primary_scores.get)

        if toxicity >= 0.70:
            risk_level = "critical"
        elif toxicity >= 0.50:
            risk_level = "high"
        elif toxicity >= 0.25:
            risk_level = "elevated"
        else:
            risk_level = "low"

        output.append(
            {
                "toxicity": round(toxicity, 4),
                "severe_toxicity": round(severe, 4),
                "obscene": round(obscene, 4),
                "identity_attack": round(identity_attack, 4),
                "insult": round(insult, 4),
                "threat": round(threat, 4),
                "sexual_explicit": round(sexual_explicit, 4),
                "strongest_signal": strongest_signal,
                "risk_level": risk_level,
            }
        )

    return output


def heuristic_emotion(texts: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []

    for text in texts:
        tokens = _tokens(text)
        total = max(len(tokens), 1)

        anger_hits = sum(token in ANGER_WORDS for token in tokens)
        disgust_hits = sum(token in DISGUST_WORDS for token in tokens)
        fear_hits = sum(token in FEAR_WORDS for token in tokens)
        joy_hits = sum(token in JOY_WORDS for token in tokens)
        sadness_hits = sum(token in SADNESS_WORDS for token in tokens)

        scores = {
            "anger": min(0.98, anger_hits / total * 3.0),
            "disgust": min(0.98, disgust_hits / total * 3.0),
            "fear": min(0.98, fear_hits / total * 3.0),
            "joy": min(0.98, joy_hits / total * 3.0),
            "neutral": 0.62,
            "sadness": min(0.98, sadness_hits / total * 3.0),
            "surprise": min(0.90, text.count("!") / total * 1.5),
        }

        negative_sum = (
            scores["anger"]
            + scores["disgust"]
            + scores["fear"]
            + scores["sadness"]
        )

        if negative_sum > 0:
            scores["neutral"] = max(0.05, 1.0 - negative_sum)
        elif scores["joy"] > 0:
            scores["neutral"] = max(0.05, 1.0 - scores["joy"])

        total_score = sum(scores.values())
        scores = {
            emotion: round(value / max(total_score, 1e-9), 4)
            for emotion, value in scores.items()
        }

        dominant = max(scores, key=scores.get)
        negative_emotion = (
            scores["anger"]
            + scores["disgust"]
            + scores["fear"]
            + scores["sadness"]
        )

        output.append(
            {
                "dominant_emotion": dominant,
                "scores": scores,
                "negative_emotion_intensity": round(negative_emotion, 4),
            }
        )

    return output
