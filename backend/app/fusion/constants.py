from __future__ import annotations

CANONICAL_LABELS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised",
)

LABEL_ALIASES = {
    "anger": "angry",
    "angry": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "happiness": "happy",
    "joy": "happy",
    "neutral": "neutral",
    "calm": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprised",
    "surprised": "surprised",
}


def normalize_emotion_label(label: str) -> str:
    raw = str(label).strip().lower().replace("_", " ").replace("-", " ")
    compact = " ".join(raw.split())
    return LABEL_ALIASES.get(compact, compact.replace(" ", "_"))
