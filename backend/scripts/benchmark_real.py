from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.evaluation.benchmark import benchmark_prediction_files
from app.evaluation.html_report import render_benchmark_html
from app.fusion.training import load_artifact


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark CEREBRO classical, deep, calibrated and fused emotion models "
            "on held-out speaker-independent prediction records."
        )
    )
    parser.add_argument("--validation", required=True, help="Validation predictions JSONL")
    parser.add_argument("--test", required=True, help="Held-out test predictions JSONL")
    parser.add_argument("--calibration-model", required=True, help="Frozen fusion calibration .joblib")
    parser.add_argument("--output", default="artifacts/evaluation/evaluation.json", help="JSON report path")
    parser.add_argument("--html", default="artifacts/evaluation/evaluation.html", help="HTML report path")
    parser.add_argument("--dataset", default="RAVDESS Speech")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    try:
        artifact = load_artifact(args.calibration_model)
        result = benchmark_prediction_files(
            args.validation,
            args.test,
            artifact,
            dataset=args.dataset,
            seed=args.seed,
            calibration_bins=args.bins,
        )
        payload = result.to_dict()
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        html_path = render_benchmark_html(payload, args.html)
        payload["json_report"] = str(output)
        payload["html_report"] = str(html_path)
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:
        print(f"BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
