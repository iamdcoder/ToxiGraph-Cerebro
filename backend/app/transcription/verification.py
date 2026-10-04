from __future__ import annotations

import re
from difflib import SequenceMatcher
from dataclasses import dataclass


_TOKEN_PATTERN = re.compile(r"[^\w']+", re.UNICODE)


@dataclass(frozen=True)
class TranscriptVerification:
    status: str
    similarity: float
    word_error_rate: float
    words_a: int
    words_b: int
    normalized_a: str
    normalized_b: str
    interpretation: str
    method: str = "normalized word edit similarity"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "similarity": self.similarity,
            "word_error_rate": self.word_error_rate,
            "words_a": self.words_a,
            "words_b": self.words_b,
            "normalized_a": self.normalized_a,
            "normalized_b": self.normalized_b,
            "interpretation": self.interpretation,
            "method": self.method,
        }


def normalize_transcript(text: str) -> str:
    lowered = str(text or "").strip().lower()
    cleaned = _TOKEN_PATTERN.sub(" ", lowered)
    return " ".join(cleaned.split())


def word_error_rate(reference: list[str], hypothesis: list[str]) -> float:
    if not reference:
        return 0.0 if not hypothesis else 1.0
    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (ref_token != hyp_token)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return min(1.0, previous[-1] / len(reference))


def verify_transcripts(text_a: str, text_b: str) -> TranscriptVerification:
    normalized_a = normalize_transcript(text_a)
    normalized_b = normalize_transcript(text_b)
    tokens_a = normalized_a.split()
    tokens_b = normalized_b.split()

    if not tokens_a or not tokens_b:
        return TranscriptVerification(
            status="uncertain",
            similarity=0.0,
            word_error_rate=1.0 if tokens_a or tokens_b else 0.0,
            words_a=len(tokens_a),
            words_b=len(tokens_b),
            normalized_a=normalized_a,
            normalized_b=normalized_b,
            interpretation=(
                "The transcription result was empty for at least one recording, so CEREBRO cannot reliably verify whether the spoken content matched."
            ),
        )

    distance = word_error_rate(tokens_a, tokens_b)
    token_similarity = float(max(0.0, 1.0 - distance))
    character_similarity = float(SequenceMatcher(None, normalized_a, normalized_b).ratio())
    similarity = float(0.80 * character_similarity + 0.20 * token_similarity)

    if similarity >= 0.85 and distance <= 0.35:
        status = "same"
        interpretation = "The transcribed content is highly similar; minor transcription differences are treated as likely ASR variation rather than a different sentence."
    elif similarity >= 0.45:
        status = "uncertain"
        interpretation = "The transcribed word sequences overlap substantially, but the difference is large enough that CEREBRO will not claim the content was identical."
    else:
        status = "different"
        interpretation = "The transcribed word sequences differ materially, so the two recordings should not be treated as a controlled same-sentence comparison."

    return TranscriptVerification(
        status=status,
        similarity=similarity,
        word_error_rate=float(distance),
        words_a=len(tokens_a),
        words_b=len(tokens_b),
        normalized_a=normalized_a,
        normalized_b=normalized_b,
        interpretation=interpretation,
    )
