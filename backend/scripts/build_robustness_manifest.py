from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.datasets.ravdess import build_ravdess_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a RAVDESS robustness manifest from the held-out speakers in a Phase-12 report.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="artifacts/robustness/manifest.jsonl")
    parser.add_argument("--evaluation-report", default="artifacts/evaluation/evaluation.json")
    parser.add_argument("--speaker", action="append", dest="speakers", help="Optional speaker override; may be repeated.")
    args = parser.parse_args()

    samples = build_ravdess_manifest(args.data_dir)
    speakers = set(args.speakers or [])
    report_path = Path(args.evaluation_report).expanduser().resolve()
    if not speakers and report_path.exists():
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        speakers = {str(item) for item in payload.get("test_speakers", [])}
    if not speakers:
        raise SystemExit("No held-out speakers found. Run Phase 12 first or pass --speaker explicitly.")

    selected = [sample for sample in samples if sample.speaker_id in speakers]
    if not selected:
        raise SystemExit(f"No RAVDESS samples found for speakers: {sorted(speakers)}")

    destination = Path(args.output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for sample in selected:
            handle.write(json.dumps({
                "audio_path": str(sample.path),
                "speaker_id": sample.speaker_id,
                "label": sample.emotion,
                "emotion_code": sample.emotion_code,
                "intensity_code": sample.intensity_code,
                "statement_code": sample.statement_code,
                "repetition_code": sample.repetition_code,
            }) + "\n")
    print(json.dumps({
        "output": str(destination),
        "records": len(selected),
        "speakers": sorted(speakers),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
