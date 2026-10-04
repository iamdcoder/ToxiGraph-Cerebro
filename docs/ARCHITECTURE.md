# CEREBRO Architecture

## Design principle

CEREBRO is organized as composable analysis layers. Audio ingestion is kept separate from feature extraction; model inference is kept separate from calibration; speaker-aware reasoning is kept separate from single-speaker inference; and historical personalization never changes the global benchmark model.

## Runtime planes

### Offline / file analysis
Audio file → validation → features/models → fusion → affect → explanations → optional ASR/comparison → report.

### Live single-speaker
Browser AudioWorklet → PCM16 WebSocket → rolling emotion windows → live state/trajectory.

### Live multi-speaker
Browser AudioWorklet → PCM16 WebSocket → bounded rolling context → diarization → persistent speaker IDs → stable turns → emotion/affect inference → interaction graph → conversation dynamics → state snapshot.

## Real-time multi-speaker design

The real-time multi-speaker layer deliberately uses rolling-context re-diarization. The default context is 8 seconds and the update hop is 1 second. The diarizer may rename local speakers between windows, so `LiveMultiSpeakerSession` matches local labels to persistent IDs using temporal-overlap ratios. Recent turns remain provisional during a stabilization horizon.

This design is near-real-time, not causal streaming diarization. The current pyannote pipeline is not a causal live diarizer. The bounded context strategy is the engineering compromise that keeps the interface responsive while allowing speaker assignments to be revised with more context.

## Concurrency

The WebSocket handler keeps receive and analysis tasks separate. Heavy model work is moved off the async event loop, and the shared multi-model inference path is protected by a process-local lock.

## State boundaries

- `audio/` owns decoding/normalization contracts.
- `features/` owns the 77-D acoustic contract.
- `emotion_baseline/`, `deep_emotion/`, `dimensional_emotion/` own model adapters.
- `fusion/` owns calibration and model combination.
- `speaker_diarization/` owns diarization model integration.
- `conversation_graph/`, `conversation_state/`, `conversation_dynamics/` own higher-level reasoning.
- `live/` and `live_multi_speaker/` own streaming session state only; they orchestrate existing engines rather than duplicating model logic.
- `longitudinal_emotion/`, `voice_change/`, `cross_session_patterns/`, and `personalization/` operate on compact profile-linked summaries.

## Security and privacy boundary

Security middleware is applied to HTTP requests, while live WebSockets enforce origin, optional token authentication, connection limits and existing message/session caps. Profile-linked persistence is separated into privacy services so deletion and retention policies do not leak into model code.

## Failure philosophy

CEREBRO prefers explicit unavailable/provisional states over fabricated predictions. Missing model artifacts, insufficient history, silent audio, poor-quality input and incomplete diarization should be represented in the result rather than hidden.
