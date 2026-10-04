import re
from typing import Any


NEGATION_WORDS = {
    "not",
    "never",
    "no",
    "don't",
    "dont",
    "doesn't",
    "doesnt",
    "didn't",
    "didnt",
    "can't",
    "cant",
    "cannot",
    "won't",
    "wont",
    "isn't",
    "isnt",
    "aren't",
    "arent",
    "wasn't",
    "wasnt",
    "weren't",
    "werent",
}

HEDGE_WORDS = {
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "might",
    "could",
    "think",
    "guess",
    "seems",
    "seem",
    "likely",
}

PROFANITY_WORDS = {
    "fuck",
    "fucking",
    "shit",
    "bitch",
    "asshole",
    "idiot",
    "moron",
    "stupid",
    "dumb",
}


EMOJI_SENTIMENT = {
    "😀": 1.0,
    "😃": 1.0,
    "😄": 1.0,
    "😁": 1.0,
    "😂": 0.8,
    "😊": 1.0,
    "❤️": 1.0,
    "❤": 1.0,
    "👍": 0.8,
    "🔥": 0.5,

    "😐": 0.0,
    "😶": 0.0,

    "😞": -0.7,
    "😔": -0.7,
    "😢": -0.8,
    "😭": -0.9,
    "😡": -1.0,
    "🤬": -1.0,
    "😠": -1.0,
    "🤢": -0.8,
    "💔": -1.0,
}


class LinguisticAnalyzer:

    def _safe_ratio(
        self,
        numerator: float,
        denominator: float,
    ) -> float:
        if denominator <= 0:
            return 0.0

        return numerator / denominator

    def analyze(
        self,
        text: str,
        parent_text: str | None = None,
    ) -> dict[str, Any]:

        clean_text = text.strip()

        words = re.findall(
            r"\b[\w']+\b",
            clean_text,
            flags=re.UNICODE,
        )

        lower_words = [
            word.lower()
            for word in words
        ]

        word_count = len(words)

        characters = [
            char
            for char in clean_text
            if char.isalpha()
        ]

        uppercase_characters = [
            char
            for char in characters
            if char.isupper()
        ]

        caps_ratio = self._safe_ratio(
            len(uppercase_characters),
            len(characters),
        )

        exclamation_density = self._safe_ratio(
            clean_text.count("!"),
            word_count,
        )

        question_density = self._safe_ratio(
            clean_text.count("?"),
            word_count,
        )

        negation_count = sum(
            word in NEGATION_WORDS
            for word in lower_words
        )

        hedge_count = sum(
            word in HEDGE_WORDS
            for word in lower_words
        )

        you_count = sum(
            word in {"you", "your", "you're", "youre", "yours"}
            for word in lower_words
        )

        self_count = sum(
            word in {"i", "me", "my", "mine", "we", "us", "our", "ours"}
            for word in lower_words
        )

        pronoun_shift = self._safe_ratio(
            you_count - self_count,
            you_count + self_count,
        )

        unique_words = len(set(lower_words))

        lexical_diversity = self._safe_ratio(
            unique_words,
            word_count,
        )

        profanity_count = sum(
            word in PROFANITY_WORDS
            for word in lower_words
        )

        profanity_score = self._safe_ratio(
            profanity_count,
            word_count,
        )

        emojis = [
            char
            for char in clean_text
            if char in EMOJI_SENTIMENT
        ]

        emoji_score = 0.0

        if emojis:
            emoji_score = sum(
                EMOJI_SENTIMENT[emoji]
                for emoji in emojis
            ) / len(emojis)

        reply_length_ratio = 1.0

        if parent_text is not None:
            parent_word_count = len(
                re.findall(
                    r"\b[\w']+\b",
                    parent_text,
                    flags=re.UNICODE,
                )
            )

            if parent_word_count > 0:
                reply_length_ratio = (
                    word_count / parent_word_count
                )

        avg_word_length = self._safe_ratio(
            sum(len(word) for word in words),
            word_count,
        )

        return {
            "word_count": word_count,
            "character_count": len(clean_text),
            "avg_word_length": round(
                avg_word_length,
                4,
            ),
            "caps_ratio": round(
                caps_ratio,
                4,
            ),
            "exclamation_density": round(
                exclamation_density,
                4,
            ),
            "question_density": round(
                question_density,
                4,
            ),
            "negation_density": round(
                self._safe_ratio(
                    negation_count,
                    word_count,
                ),
                4,
            ),
            "hedging_score": round(
                self._safe_ratio(
                    hedge_count,
                    word_count,
                ),
                4,
            ),
            "pronoun_shift": round(
                pronoun_shift,
                4,
            ),
            "lexical_diversity": round(
                lexical_diversity,
                4,
            ),
            "profanity_score": round(
                profanity_score,
                4,
            ),
            "emoji_sentiment": round(
                emoji_score,
                4,
            ),
            "reply_length_ratio": round(
                reply_length_ratio,
                4,
            ),
        }