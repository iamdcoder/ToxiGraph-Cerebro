from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from app.audio.processor import AudioProcessor
from app.deep_emotion.service import get_model as get_deep_model
from app.emotion_baseline.model import ClassicalEmotionModel
from app.features.acoustic import extract_acoustic_features
from app.fusion.service import get_engine
from app.robustness.benchmark import run_robustness_benchmark
from app.robustness.html_report import render_robustness_html


def _load_manifest(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Manifest line {line_number} is not an object.")
            rows.append(payload)
    if not rows:
        raise ValueError("Robustness manifest contains no records.")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the final CEREBRO fused emotion system through controlled audio stress tests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default="artifacts/robustness/robustness.json")
    parser.add_argument("--html", default="artifacts/robustness/robustness.html")
    parser.add_argument("--dataset", default="RAVDESS Speech")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--condition", action="append", dest="conditions")
    args = parser.parse_args()

    try:
        records = _load_manifest(args.manifest)
        classical = ClassicalEmotionModel.load(__import__("app.config", fromlist=["settings"]).settings.CLASSICAL_MODEL_PATH)
        deep = get_deep_model()
        fusion = get_engine()
        processor = AudioProcessor()

        def predictor(samples, sample_rate):
            features = extract_acoustic_features(samples, sample_rate)
            classical_prediction = classical.predict(features.to_vector())
            deep_prediction = deep.predict(samples, sample_rate)
            combined = fusion.combine(classical_prediction["probabilities"], deep_prediction.probabilities)
            return combined.probabilities

        result = run_robustness_benchmark(
            records,
            predictor,
            dataset=args.dataset,
            seed=args.seed,
            conditions=args.conditions,
        )
        payload = result.to_dict()
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        html_path = render_robustness_html(payload, args.html)
        payload["json_report"] = str(output)
        payload["html_report"] = str(html_path)
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:
        print(f"ROBUSTNESS BENCHMARK FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
