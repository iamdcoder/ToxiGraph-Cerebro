# CEREBRO Evaluation

## Evaluation principles

CEREBRO separates three kinds of verification:

1. **Software correctness** — unit/integration tests and API contracts.
2. **Model performance** — speaker-independent evaluation on real held-out data.
3. **Robustness** — controlled perturbation testing to measure degradation.

Synthetic fixtures can validate algorithms and integrations, but they must never be presented as model-performance results.

## Speaker-independent protocol

The RAVDESS training pipeline parses actor identity from filenames and keeps speakers separated across train, validation and test. Phase 25 uses grouped cross-validation by speaker on the training portion. Test speakers remain untouched until final evaluation.

## Metrics

Classification:
- accuracy
- balanced accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- per-class F1
- confusion matrix

Calibration:
- negative log likelihood
- Brier score
- expected calibration error

Model analysis:
- raw vs calibrated
- classical vs deep
- fused vs individual branches
- speaker-level results

## Real-data workflow

```bash
cd backend
python -m scripts.optimize_baseline --data-dir PATH_TO_RAVDESS --output-model models/cerebro_classical_emotion_optimized.joblib --output-dir artifacts/optimization/classical --feature-cache artifacts/cache/ravdess_acoustic_features.npz --cv-folds 4

python -m scripts.generate_fusion_predictions --data-dir PATH_TO_RAVDESS --classical-model models/cerebro_classical_emotion_optimized.joblib --output-dir artifacts/fusion_predictions_optimized

python -m scripts.train_fusion --validation artifacts/fusion_predictions_optimized/validation.jsonl --test artifacts/fusion_predictions_optimized/test.jsonl --output-model models/cerebro_fusion_calibration_optimized.joblib

python -m scripts.benchmark_real --validation artifacts/fusion_predictions_optimized/validation.jsonl --test artifacts/fusion_predictions_optimized/test.jsonl --calibration-model models/cerebro_fusion_calibration_optimized.joblib
```

## Robustness

Phase 13 provides controlled perturbations such as gain changes, additive noise, clipping, speed changes, bandwidth reduction, reverb and quantization. It measures performance degradation relative to clean audio.

## Reporting rule

If real training/evaluation artifacts are absent, the UI should say so. Do not add manually typed accuracy/F1 values to the README or dashboard.

## Current software gate

The submission release contains **271 passing automated backend tests**. Browser microphone access and the production frontend build still require local environment verification when dependencies and a physical microphone are available.
