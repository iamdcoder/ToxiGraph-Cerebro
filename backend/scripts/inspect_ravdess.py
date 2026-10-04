from __future__ import annotations

import argparse
import json

from app.datasets.ravdess import build_ravdess_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect and validate an extracted RAVDESS Speech directory.")
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    samples = build_ravdess_manifest(args.data_dir)
    speakers = sorted({sample.speaker_id for sample in samples})
    labels = sorted({sample.emotion for sample in samples})
    print(json.dumps({
        "valid": True,
        "samples": len(samples),
        "speakers": speakers,
        "speaker_count": len(speakers),
        "emotions": labels,
        "emotion_count": len(labels),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
