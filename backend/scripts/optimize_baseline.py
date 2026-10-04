from __future__ import annotations

import argparse
import json
import sys

from app.emotion_baseline.training import SEED
from app.optimization.classical import optimize_classical_baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune CEREBRO's classical acoustic emotion model with speaker-grouped cross-validation."
    )
    parser.add_argument("--data-dir", required=True, help="Extracted RAVDESS Speech directory")
    parser.add_argument(
        "--output-model",
        default="models/cerebro_classical_emotion_optimized.joblib",
        help="Destination for the optimized model artifact",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/optimization/classical",
        help="Directory for optimization.json and training_metadata.json",
    )
    parser.add_argument("--feature-cache", default="artifacts/cache/ravdess_acoustic_features.npz")
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    try:
        result = optimize_classical_baseline(
            args.data_dir,
            args.output_model,
            args.output_dir,
            seed=args.seed,
            cv_folds=args.cv_folds,
            feature_cache=args.feature_cache,
        )
    except Exception as exc:
        print(f"OPTIMIZATION FAILED: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
