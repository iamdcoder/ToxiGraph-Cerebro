from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.emotion_baseline.training import SEED, train_baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the CEREBRO classical speech-emotion baseline on RAVDESS Speech."
    )
    parser.add_argument("--data-dir", required=True, help="Path to an extracted RAVDESS Speech directory.")
    parser.add_argument(
        "--output-model",
        default="models/cerebro_classical_emotion.joblib",
        help="Destination for the trained model artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/baseline",
        help="Directory for evaluation.json, training_metadata.json and manifest.csv.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    try:
        result = train_baseline(args.data_dir, args.output_model, args.output_dir, seed=args.seed)
    except Exception as exc:
        print(f"TRAINING FAILED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
