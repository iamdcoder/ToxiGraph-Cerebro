from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from app.audio.processor import AudioProcessor
from app.deep_emotion.service import clear_model_cache, get_model as get_deep_model
from app.datasets.ravdess import RAVDESSSample, build_ravdess_manifest
from app.emotion_baseline.model import ClassicalEmotionModel
from app.emotion_baseline.training import speaker_independent_split
from app.features.acoustic import extract_acoustic_features
from app.fusion.constants import CANONICAL_LABELS


def _predict_samples(
    samples: Iterable[RAVDESSSample],
    classical_model: ClassicalEmotionModel,
    deep_model,
) -> list[dict]:
    processor = AudioProcessor()
    records: list[dict] = []
    for sample in samples:
        processed = processor.process(sample.path.read_bytes())
        if processed.is_silent:
            raise ValueError(f"Cannot create fusion prediction for silent sample: {sample.path}")
        features = extract_acoustic_features(processed.samples, processed.sample_rate)
        classical = classical_model.predict(features.to_vector())
        deep = deep_model.predict(processed.samples, processed.sample_rate)
        records.append(
            {
                "path": str(sample.path),
                "speaker_id": sample.speaker_id,
                "label": sample.emotion,
                "classical_model": classical["model_name"],
                "deep_model": deep.metadata.model_id,
                "classical_probabilities": classical["probabilities"],
                "deep_probabilities": deep.probabilities,
            }
        )
    return records


def _write_jsonl(path: Path, records: list[dict], split_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            payload = {"split": split_name, **record}
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate speaker-independent classical/deep emotion probabilities for CEREBRO fusion calibration."
    )
    parser.add_argument("--data-dir", required=True, help="Extracted RAVDESS Speech directory")
    parser.add_argument(
        "--classical-model",
        default="models/cerebro_classical_emotion.joblib",
        help="Trained Phase-3 classical model artifact",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/fusion_predictions",
        help="Directory for validation.jsonl and test.jsonl",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = build_ravdess_manifest(args.data_dir)
    split = speaker_independent_split(samples, seed=args.seed)
    classical_model = ClassicalEmotionModel.load(args.classical_model)
    clear_model_cache()
    deep_model = get_deep_model()

    output_dir = Path(args.output_dir).expanduser().resolve()
    validation = _predict_samples(split.validation, classical_model, deep_model)
    test = _predict_samples(split.test, classical_model, deep_model)
    _write_jsonl(output_dir / "validation.jsonl", validation, "validation")
    _write_jsonl(output_dir / "test.jsonl", test, "test")

    print(
        json.dumps(
            {
                "success": True,
                "canonical_labels": list(CANONICAL_LABELS),
                "validation_records": len(validation),
                "test_records": len(test),
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
