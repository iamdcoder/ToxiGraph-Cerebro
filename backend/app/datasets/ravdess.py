from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RAVDESS_EMOTIONS: dict[int, str] = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

@dataclass(frozen=True)
class RAVDESSSample:
    path: Path
    speaker_id: str
    emotion: str
    emotion_code: int
    intensity_code: int
    statement_code: int
    repetition_code: int


def parse_ravdess_filename(path: str | Path) -> RAVDESSSample:
    file_path = Path(path)
    if file_path.suffix.lower() != ".wav":
        raise ValueError("RAVDESS sample must be a .wav file.")

    parts = file_path.stem.split("-")
    if len(parts) != 7:
        raise ValueError(
            f"Invalid RAVDESS filename: {file_path.name}. Expected 7 hyphen-separated fields."
        )

    try:
        modality, vocal_channel, emotion_code, intensity_code, statement_code, repetition_code, actor_code = (
            int(value) for value in parts
        )
    except ValueError as exc:
        raise ValueError(f"Invalid numeric field in RAVDESS filename: {file_path.name}.") from exc

    if modality != 3:
        raise ValueError("Only RAVDESS audio-only files (modality 03) are supported.")
    if vocal_channel != 1:
        raise ValueError("Only RAVDESS speech files (vocal channel 01) are supported.")
    if emotion_code not in RAVDESS_EMOTIONS:
        raise ValueError(f"Unknown RAVDESS emotion code: {emotion_code}.")
    if not 1 <= intensity_code <= 2:
        raise ValueError(f"Invalid RAVDESS intensity code: {intensity_code}.")
    if not 1 <= statement_code <= 2:
        raise ValueError(f"Invalid RAVDESS statement code: {statement_code}.")
    if not 1 <= repetition_code <= 2:
        raise ValueError(f"Invalid RAVDESS repetition code: {repetition_code}.")
    if not 1 <= actor_code <= 24:
        raise ValueError(f"Invalid RAVDESS actor code: {actor_code}.")

    return RAVDESSSample(
        path=file_path,
        speaker_id=f"actor_{actor_code:02d}",
        emotion=RAVDESS_EMOTIONS[emotion_code],
        emotion_code=emotion_code,
        intensity_code=intensity_code,
        statement_code=statement_code,
        repetition_code=repetition_code,
    )


def build_ravdess_manifest(data_dir: str | Path) -> list[RAVDESSSample]:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"RAVDESS data directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"RAVDESS data path is not a directory: {root}")

    samples: list[RAVDESSSample] = []
    invalid: list[str] = []
    for path in sorted(root.rglob("*.wav")):
        try:
            samples.append(parse_ravdess_filename(path))
        except ValueError as exc:
            invalid.append(str(exc))

    if not samples:
        detail = invalid[0] if invalid else "No WAV files were found."
        raise ValueError(f"No valid RAVDESS speech samples found. {detail}")

    if invalid:
        raise ValueError(
            f"Found {len(invalid)} invalid WAV filenames. First issue: {invalid[0]}"
        )

    speakers = {sample.speaker_id for sample in samples}
    labels = {sample.emotion for sample in samples}
    if len(speakers) < 6:
        raise ValueError("At least 6 distinct speakers are required for a speaker-independent baseline.")
    if len(labels) < 4:
        raise ValueError("At least 4 emotion classes are required for the baseline.")

    return samples
