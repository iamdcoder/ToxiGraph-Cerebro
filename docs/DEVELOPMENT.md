# CEREBRO Development Guide

## Phase discipline

Every phase follows:

```text
Implement → Unit test → Integration test → Manual/environment check → Freeze → Next phase
```

A phase is not considered green because a single path works. The complete regression suite must pass.

## Code boundaries

Keep DSP, model adapters, inference orchestration, and UI components separated. Reuse the existing production pipeline rather than implementing duplicate model logic inside individual panels.

## Tests

```bash
cd backend
pytest -q
```

Add focused tests for every new service/router and at least one integration path for each new public feature.

## Model artifacts

Never commit large pretrained weights, trained joblib artifacts, datasets or personal recordings. Put artifacts under ignored paths and document the command required to regenerate them.

## Personalization

Personal baseline, longitudinal history and personalized calibration must remain separate from the global model benchmark. Explicit user feedback may adapt a speaker profile, but it must never silently change the benchmark training data.

## Real-time behavior

Do not block the async event loop with model loading or CPU-heavy inference. Use worker threads/tasks where appropriate, cap durations/chunk sizes, and send explicit error/provisional states over WebSocket.

## Product wording

Use language such as “emotion conveyed through speech”, “expressed affect”, “model confidence” and “observed interaction”. Avoid claims that CEREBRO can read hidden mental states, diagnose a person, or prove causal emotional influence between speakers.

## Release hygiene

Before packaging:

- run full tests
- compile Python
- validate package metadata
- parse changed frontend source
- remove caches
- confirm no datasets/model weights/personal audio are present
- update durable technical documentation
- verify version consistency
