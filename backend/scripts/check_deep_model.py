from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.audio.processor import AudioProcessor
from app.config import settings
from app.deep_emotion.service import get_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Check and run the configured CEREBRO deep speech emotion model.")
    parser.add_argument("audio", type=Path, help="Path to a PCM WAV recording.")
    args = parser.parse_args()

    processed = AudioProcessor().process(args.audio.read_bytes())
    if processed.is_silent:
        raise SystemExit("Audio is nearly silent.")

    model = get_model()
    prediction = model.predict(processed.samples, processed.sample_rate)
    print(json.dumps({
        "model": model.metadata(),
        "prediction": {
            "emotion": prediction.emotion,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
            "chunks_analyzed": prediction.chunks_analyzed,
        },
        "source": settings.DEEP_EMOTION_MODEL,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
