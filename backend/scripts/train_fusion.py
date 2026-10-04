from __future__ import annotations

import argparse
import json

from app.fusion.training import (
    evaluate_fusion_records,
    fit_fusion_calibrator,
    load_prediction_records,
    save_artifact,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit CEREBRO model calibration and probability fusion.")
    parser.add_argument("--validation", required=True, help="JSON/JSONL validation prediction records")
    parser.add_argument("--test", help="Optional held-out JSON/JSONL test prediction records")
    parser.add_argument("--output-model", required=True, help="Output .joblib calibration artifact")
    args = parser.parse_args()

    validation_records = load_prediction_records(args.validation)
    artifact = fit_fusion_calibrator(validation_records)
    payload = artifact.to_dict()

    if args.test:
        test_records = load_prediction_records(args.test)
        payload["held_out_test"] = evaluate_fusion_records(test_records, artifact)

    save_artifact(artifact, args.output_model)
    output_path = __import__("pathlib").Path(args.output_model).expanduser().resolve()
    if "held_out_test" in payload:
        evaluation_path = output_path.parent / "fusion_evaluation.json"
        evaluation_path.write_text(json.dumps({"success": True, "artifact": payload}, indent=2), encoding="utf-8")
        payload["held_out_evaluation_path"] = str(evaluation_path)
    print(json.dumps({"success": True, "artifact": payload}, indent=2))


if __name__ == "__main__":
    main()
