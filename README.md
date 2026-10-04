# CEREBRO — Speech Emotion Intelligence

> **ALGOTHON'26 Submission · Algoxilla · Online Hackathon · Individual Build**
>
> CEREBRO is an end-to-end speech-emotion intelligence platform that analyzes **how speech is expressed**, not only what was said. It combines acoustic signal processing, deep speech representations, calibrated model fusion, continuous affect estimation, temporal dynamics, uncertainty, personalization, speaker-aware conversation intelligence, real-time multi-speaker analysis, evaluation, robustness testing, and privacy/security controls in one extensible system.

![Backend](https://img.shields.io/badge/backend-FastAPI-0f172a?style=flat-square)
![Frontend](https://img.shields.io/badge/frontend-React%2019-0f172a?style=flat-square)
![Audio](https://img.shields.io/badge/audio-16%20kHz%20PCM16-0f172a?style=flat-square)
![Tests](https://img.shields.io/badge/tests-271%20passing-0f172a?style=flat-square)
![Version](https://img.shields.io/badge/version-0.27.0-0f172a?style=flat-square)

---

## 1. Project at a glance

A sentence does not determine how it was expressed.

The same words can be spoken calmly, angrily, nervously, sadly, enthusiastically, or with different levels of activation and intensity. CEREBRO is built to model **emotion conveyed through the voice** using measurable acoustic evidence and learned speech representations.

A typical single-speaker result can look conceptually like:

```text
EMOTION
Angry                         81.4%

CONTINUOUS AFFECT
Valence                      -0.62
Arousal                      +0.84
Dominance                    +0.67

VOICE EVIDENCE
• elevated vocal energy
• increased pitch variation
• strong acoustic activation

MODEL AGREEMENT
High

UNCERTAINTY
Moderate
```

For longer speech, CEREBRO tracks emotion and affect through time. For multi-speaker recordings, it separates speakers, builds directed interaction graphs, measures response-to-response affect changes, and estimates conversation states. For repeat users, it can build a personal acoustic baseline, maintain compact session history, detect unusual vocal changes, discover recurring patterns, and apply explicit-feedback personalization.

The project deliberately avoids claiming to read a person's private internal emotional state. Its outputs are framed as **expressed/conveyed emotion, affective estimates, model confidence, and observed interaction patterns**.

---

## 2. ALGOTHON'26

CEREBRO was built as the project's submission for **ALGOTHON'26 by Algoxilla**.

The public event listing describes ALGOTHON'26 as an online, student-driven coding hackathon. The official Devfolio schedule lists the hackathon start on **4 October 2026** and the event is online. The current Unstop listing explicitly allows individual participation, which matches this repository's individual submission format.

Official event references:

- [ALGOTHON'26 — Devfolio](https://algothon.devfolio.co/)
- [ALGOTHON'26 — Schedule](https://algothon.devfolio.co/schedule)
- [ALGOTHON'26 — Unstop](https://unstop.com/hackathons/algothon26-algoxilla-1760293/amp)

This repository is intended to be the **source-code submission package**. It intentionally does not bundle:

- private audio recordings
- raw transcripts from private sessions
- authentication secrets
- API keys
- large pretrained model weights
- downloaded benchmark datasets
- generated personal profile data

---

# 3. The problem

Most simple emotion systems treat emotion recognition as a single classification problem:

```text
Audio → model → emotion
```

That is useful, but insufficient for realistic voice analysis.

A serious speech-emotion system also has to deal with:

1. **Acoustic variation** — different people naturally have different pitch, energy, pace, and speaking styles.
2. **Model disagreement** — different models can produce different predictions for the same recording.
3. **Temporal variation** — emotion can change during one sentence or one conversation.
4. **Recording quality** — clipping, silence, low level, noise and bandwidth can make predictions unreliable.
5. **Multiple speakers** — conversations require speaker separation and turn-aware analysis.
6. **Personal variation** — a vocal pattern that is unusual for one person may be completely normal for another.
7. **Longitudinal variation** — a speaker's expressed vocal profile can shift between sessions.
8. **Uncertainty** — the system must be allowed to say that the evidence is weak.
9. **Privacy** — voice data and personal history should not silently become permanent application data.
10. **Evaluation** — performance must be measured on unseen speakers rather than demonstrated only with hand-picked examples.

CEREBRO is designed around those problems rather than treating emotion recognition as one isolated classifier.

---

# 4. Design philosophy

The system follows five engineering principles.

### Evidence before interpretation

A prediction should have measurable support: acoustic features, model agreement, temporal behavior, recording quality, and calibrated probabilities.

### Separate global intelligence from personalization

The global model is evaluated independently. Personal baselines, user feedback, and historical summaries never silently alter the global benchmark set.

### Uncertainty is a first-class output

The system should not force a confident label when the signal quality or model agreement is poor.

### Real-time behavior must be explicit about limitations

The multi-speaker live path uses rolling-context re-diarization. Recent speaker identities and boundaries may be revised as additional context arrives; recent turns are therefore treated as provisional when appropriate.

### Privacy by default

The profile services store compact derived summaries rather than raw microphone recordings and raw ASR transcripts. Profile deletion and retention controls are explicit.

---

# 5. System architecture

```text
                                         CEREBRO
                                            │
                                  Browser microphone
                                            │
                                   AudioWorklet / WAV
                                            │
                              PCM → mono → 16 kHz audio
                                            │
                      ┌─────────────────────┴─────────────────────┐
                      │                                           │
               Audio quality gate                         Analysis pipeline
                      │                                           │
             reliability context                  ┌───────────────┴───────────────┐
                                                 │                               │
                                        Acoustic branch                   Waveform branch
                                                 │                               │
                                           77-D features                 Deep speech encoder
                                                 │                               │
                                           Classical SER                    Deep SER
                                                 │                               │
                                                 └───────────────┬───────────────┘
                                                                 │
                                                          Calibration + fusion
                                                                 │
                                  ┌──────────────────────────────┼──────────────────────────────┐
                                  │                              │                              │
                               Emotion                         V/A/D                       Uncertainty
                                  │                              │                              │
                                  └──────────────────────────────┼──────────────────────────────┘
                                                                 │
                                                        Temporal analysis
                                                                 │
                                                       Explanation engine
                                                                 │
                      ┌──────────────────────────────────────────┼─────────────────────────────────────┐
                      │                    │                     │                │                    │
                ASR/content        Comparison Lab        Personal baseline   Speaker analysis    Longitudinal history
                verification                              │                │                    │
                      │                                   │                │                    ├─ anomaly detection
                      │                                   │                ├─ interaction graph  ├─ pattern discovery
                      │                                   │                ├─ dynamics           └─ personalization
                      │                                   │                └─ state engine
                      │                                   │
                      └───────────────────────────────────┴───────────────────────────────┐
                                                                                           │
                                                                             Real-time streaming
                                                                                           │
                                                                                 Live multi-speaker
                                                                                           │
                                                                                     Privacy layer
                                                                                           │
                                                                                      CEREBRO UI
```

---

# 6. End-to-end inference flow

## Single recording

```text
Microphone / WAV
      ↓
Audio validation
      ↓
Mono + 16 kHz normalization
      ↓
Audio quality analysis
      ↓
Acoustic feature extraction
      ↓
Classical emotion model ──────────┐
                                  ├──→ calibration → fusion → final emotion
Deep speech emotion model ────────┘
                                  │
                                  ├──→ uncertainty
                                  ├──→ explanation
                                  ├──→ V/A/D affect
                                  └──→ temporal trajectory
```

## Multi-speaker recording

```text
Conversation WAV
      ↓
Speaker diarization
      ↓
Speaker turns
      ↓
Per-speaker inference
      ↓
Persistent speaker IDs
      ↓
Interaction graph
      ↓
Conversation dynamics
      ↓
Conversation state
```

## Real-time multi-speaker

```text
Browser AudioWorklet
      ↓
PCM16 WebSocket
      ↓
8-second rolling context
      ↓
1-second update cadence
      ↓
Diarization + identity stabilization
      ↓
Emotion / V/A/D per speaker turn
      ↓
Graph + dynamics + state snapshots
      ↓
Live UI
```

---

# 7. Core intelligence layers

## 7.1 Audio ingestion

The browser records microphone audio using an `AudioWorklet` and sends PCM data. The backend validates supported WAV input, converts it to mono floating-point samples, and normalizes it to a 16 kHz target sample rate.

The ingestion layer explicitly rejects or flags:

- malformed audio
- unsupported encoding
- empty input
- excessive duration
- excessive size
- unusable silence

---

## 7.2 Audio quality engine

Emotion inference should not silently ignore the quality of the input signal.

CEREBRO measures:

- speech coverage
- silence ratio
- RMS level
- clipping
- DC offset
- dynamic range
- estimated SNR proxy
- recording duration

The result is classified as:

```text
GOOD
ACCEPTABLE
POOR
UNRELIABLE
```

The quality result contributes to downstream uncertainty rather than pretending that all recordings are equally trustworthy.

The SNR output is explicitly an **estimated signal-quality proxy**, not a laboratory-grade acoustic SNR measurement.

---

## 7.3 Acoustic feature engine

The classical branch uses a **77-dimensional acoustic representation**.

It includes families such as:

- fundamental frequency / pitch statistics
- RMS energy
- pitch variation
- spectral centroid
- spectral bandwidth
- spectral rolloff
- spectral flatness
- zero-crossing rate
- spectral contrast
- chroma
- MFCC statistics
- duration
- silence and pause structure
- speech duration
- estimated syllable rate

The feature extractor is deterministic for the same normalized signal and is isolated from model training so it can be benchmarked independently.

---

## 7.4 Classical speech-emotion model

The baseline branch supports:

- Logistic Regression
- Linear SVM
- Random Forest

The optimization pipeline uses grouped cross-validation so speakers do not leak between folds.

The selection objective is validation Macro-F1, followed by deterministic tie-breaking.

The benchmark test set remains untouched until final evaluation.

---

## 7.5 Deep speech-emotion model

The deep branch uses a configurable Hugging Face Transformers audio-classification checkpoint.

Default checkpoint:

[`Dpngtm/wav2vec2-emotion-recognition`](https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition)

The upstream model card describes it as a Wav2Vec2 speech-emotion classifier and reports seven emotion classes after merging calm and neutral. CEREBRO treats upstream benchmark numbers as **upstream numbers**, not as CEREBRO benchmark results.

The repository contains the loader and inference adapter, not the pretrained weights.

---

## 7.6 Calibration and model fusion

The classical and deep model outputs are normalized to the same canonical label space, then calibrated and fused.

```text
Classical probabilities ─────┐
                             ├──→ calibrated probabilities ─→ fusion
Deep probabilities ──────────┘                             │
                                                          ├─ final emotion
                                                          ├─ confidence
                                                          └─ model agreement
```

CEREBRO also computes model disagreement so that conflicting evidence remains visible.

Calibration and fusion parameters are fit on validation predictions only.

---

# 8. Emotion representation

CEREBRO supports two complementary representations.

## Categorical emotion

The main seven-class space is:

```text
anger
 disgust
fear
happy
neutral
sadness
surprise
```

The implementation normalizes legacy/model naming variants into a consistent internal representation.

## Continuous affect

A recording can additionally be represented using:

### Valence

Negative ↔ positive expressed affective direction.

### Arousal

Calm ↔ activated expressed state.

### Dominance

Lower ↔ higher perceived control/agency in the expressed vocal signal.

These values are **speech-affect estimates**, not measurements of a speaker's private psychological state.

Default dimensional checkpoint:

[`3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes`](https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes)

---

# 9. Temporal intelligence

A recording is rarely emotionally uniform.

CEREBRO analyzes overlapping windows and creates a readable, non-overlapping trajectory for visualization.

Example:

```text
00–01s   neutral
01–02s   neutral
02–03s   concerned
03–04s   frustrated
04–05s   angry
05–06s   angry
06–07s   angry
07–08s   cooling
```

Silent windows remain unclassified instead of being forced into `neutral`.

The temporal layer computes:

- emotion persistence
- transitions
- valence movement
- arousal movement
- dominance movement
- trajectory velocity
- volatility
- stability
- peak states

---

# 10. Explainability and uncertainty

CEREBRO uses an evidence-oriented explanation layer.

It considers:

- final emotion probabilities
- top-class margin
- predictive entropy
- classical/deep agreement
- acoustic feature evidence
- audio quality
- temporal movement
- personal-baseline deviations when available

An explanation is phrased as evidence such as:

```text
The speech contains acoustic patterns associated with the model's
anger prediction. The result is less certain because the two models
show moderate disagreement and the recording quality is acceptable
rather than high.
```

The system does **not** generate explanations such as “you are angry because…” because that would overstate what the model can establish.

---

# 11. Personal voice intelligence

## 11.1 Personal acoustic baseline

A user can optionally establish a personal reference from several natural recordings.

The baseline captures an individual's normal acoustic range so future sessions can be compared against the speaker's own history rather than against population-level absolute values.

The baseline does not retrain the global model.

## 11.2 Longitudinal history

CEREBRO can store compact summaries such as:

- session timestamp
- duration
- dominant emotion/confidence
- selected acoustic statistics
- V/A/D values
- audio-quality score
- baseline deviation context

The history is bounded and duplicate session IDs replace previous summaries.

## 11.3 Voice change and anomaly detection

The current session is compared against historical robust statistics using median/MAD-style normalization.

The engine can return:

```text
NORMAL
MINOR CHANGE
SIGNIFICANT CHANGE
UNUSUAL
```

With insufficient history, it returns **insufficient data** rather than manufacturing a baseline.

## 11.4 Cross-session pattern discovery

With enough sessions, CEREBRO searches for recurring descriptive patterns such as:

- recurring emotion distributions
- repeated emotion transitions
- recurring affect directions
- stability/volatility patterns
- repeated acoustic-affect associations

A minimum-evidence gate prevents weak histories from being described as strong patterns.

## 11.5 Explicit-feedback personalization

A user may explicitly label the emotion they intended to convey. After enough diverse, good-quality feedback, CEREBRO can learn a speaker-specific calibration layer.

This layer sits **after** the global model and cannot silently contaminate the benchmark training data.

---

# 12. Recording Comparison Lab

The Comparison Lab is built around a controlled experiment:

> **Say the same sentence twice and change the delivery.**

CEREBRO compares:

- categorical emotion
- confidence
- acoustic features
- probability distributions
- model agreement
- V/A/D movement
- recording quality

Phase 10 adds local ASR/content verification so the system can determine whether the linguistic content was approximately the same.

The intended experiment is:

```text
Same / similar words
        +
Different vocal delivery
        ↓
Different acoustic signal
        ↓
Potentially different expressed-affect prediction
```

The comparison engine does not claim that two recordings have identical text unless ASR/content verification supports that conclusion.

---

# 13. Multi-speaker conversation intelligence

## Speaker-aware analysis

For a multi-speaker recording, CEREBRO produces per-speaker summaries:

- speaker ID
- speaking time/share
- number of turns
- dominant emotion
- confidence
- V/A/D statistics
- turn-level affect

## Speaker interaction graph

The system creates directed response edges:

```text
Speaker A ─────────→ Speaker B
        4 observed responses
        median response gap: 0.8 s
        affect shifts: 2
```

The edge means **A spoke and B subsequently responded**. It is not a causal relationship.

## Conversation dynamics

Signals include:

- response latency
- affect changes
- affect synchrony
- volatility
- stability
- escalation/de-escalation indicators
- tension trend

## Conversation state engine

These low-level observations are converted into interpretable states:

```text
CALM
ENGAGED
STABLE
TENSE
ESCALATING
PEAK TENSION
COOLING
DE-ESCALATING
```

Hysteresis and transition thresholds reduce state flicker caused by a single noisy turn.

---

# 14. Real-time multi-speaker intelligence

CEREBRO supports two live modes:

### Live single-speaker

```text
Browser → WebSocket → rolling emotion inference → live affect/state
```

### Live multi-speaker

```text
Browser
  ↓
AudioWorklet
  ↓
PCM16 WebSocket
  ↓
8-second rolling context
  ↓
Diarization
  ↓
Speaker-ID stabilization
  ↓
Per-speaker inference
  ↓
Interaction graph
  ↓
Conversation dynamics
  ↓
Conversation state
  ↓
Live UI snapshot
```

The live multi-speaker context defaults to:

```text
Context:        8 s
Update cadence: 1 s
Stabilization:  0.8 s
Max session:    180 s
```

The system treats recent speaker turns as **provisional** when the rolling context can still change the diarization result.

Default diarization model:

[`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1)

The current upstream model card documents local speaker diarization from mono 16 kHz audio and requires accepting the model's access conditions and using a Hugging Face token for first-time model access.

---

# 15. Robustness and real-world stress testing

The project includes a robustness lab that applies controlled perturbations to evaluation audio, including:

- gain changes
- additive noise
- clipping
- speech-speed changes
- bandwidth reduction
- reverberation
- quantization

The report compares the perturbed condition with the clean condition using measures such as:

- accuracy
- balanced accuracy
- Macro-F1
- Weighted-F1
- confidence
- clean-vs-perturbed prediction agreement
- degradation in Macro-F1
- quality distribution

The SNR and quality indicators are interpreted as engineering diagnostics, not as clinical or psychological measurements.

---

# 16. Evaluation methodology

CEREBRO has a separate evaluation framework so the application does not confuse “the UI produced a label” with “the model is accurate.”

## Speaker-independent evaluation

The benchmark data is split by speaker identity.

```text
Train speakers
      ↓
Validation speakers
      ↓
Held-out test speakers
```

A speaker cannot appear in both validation and test.

## Metrics

The benchmark framework supports:

- Accuracy
- Balanced Accuracy
- Precision
- Recall
- Macro-F1
- Weighted-F1
- Negative Log-Likelihood
- Brier score
- Expected Calibration Error
- confusion matrices
- per-class metrics
- speaker-level summaries

## Model comparison

The intended comparison is:

```text
Classical acoustic model
        vs
Deep speech model
        vs
Calibrated/fused model
```

The same held-out test protocol is used for all relevant comparisons.

## No fabricated metrics

This source package does **not** contain invented benchmark scores.

If trained artifacts or evaluation files are absent, the UI shows that results are unavailable rather than substituting placeholder numbers.

The repository's automated tests use deterministic fixtures where necessary to validate engineering logic. Fixture results are not presented as scientific model performance.

---

# 17. Training and optimization

The classical model is reproducibly trainable from RAVDESS-style audio metadata.

### Inspect the dataset

```bash
cd backend
python -m scripts.inspect_ravdess --data-dir PATH_TO_RAVDESS
```

### Train a classical baseline

```bash
python -m scripts.train_baseline \
  --data-dir PATH_TO_RAVDESS \
  --output-model models/cerebro_classical_emotion.joblib \
  --output-dir artifacts/baseline
```

### Optimize the classical model

```bash
python -m scripts.optimize_baseline \
  --data-dir PATH_TO_RAVDESS \
  --output-model models/cerebro_classical_emotion_optimized.joblib \
  --output-dir artifacts/optimization/classical \
  --feature-cache artifacts/cache/ravdess_acoustic_features.npz \
  --cv-folds 4
```

### Generate fusion predictions

```bash
python -m scripts.generate_fusion_predictions \
  --data-dir PATH_TO_RAVDESS \
  --classical-model models/cerebro_classical_emotion.joblib \
  --output-dir artifacts/fusion_predictions
```

### Train the fusion calibration artifact

```bash
python -m scripts.train_fusion \
  --validation artifacts/fusion_predictions/validation.jsonl \
  --test artifacts/fusion_predictions/test.jsonl \
  --output-model models/cerebro_fusion_calibration.joblib
```

The deep model and dimensional affect model are loaded separately from their configured Hugging Face checkpoints; no model weights are committed to this repository.

---

# 18. Data and model provenance

## RAVDESS

CEREBRO's classical training/evaluation tooling supports the RAVDESS speech-emotion dataset.

- Official dataset page: <https://zenodo.org/records/1188976>

The actual dataset should be downloaded separately and never committed to this repository.

## Deep emotion checkpoint

- `Dpngtm/wav2vec2-emotion-recognition`
- <https://huggingface.co/Dpngtm/wav2vec2-emotion-recognition>

## Dimensional affect checkpoint

- `3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes`
- <https://huggingface.co/3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes>

## ASR

- Default: `openai/whisper-small`
- <https://huggingface.co/openai/whisper-small>

## Speaker diarization

- Default: `pyannote/speaker-diarization-community-1`
- <https://huggingface.co/pyannote/speaker-diarization-community-1>

Always review the current upstream model card and dataset terms before redistribution or production deployment.

---

# 19. Privacy and security

Privacy is part of the application architecture rather than an afterthought.

## Raw audio

The profile/history services do not persist raw microphone recordings.

## Transcripts

The profile/history services do not persist raw ASR transcripts.

## Derived profile data

Persistent services may store compact derived summaries under configurable directories:

```text
PERSONAL_BASELINE_DIR
VOICE_HISTORY_DIR
PERSONALIZATION_DIR
```

## Retention

The default retention setting is:

```env
PRIVACY_RETENTION_ENABLED=true
PRIVACY_RETENTION_DAYS=30
```

Expired profile JSON files are pruned at application initialization.

## Delete everything for a profile

```text
DELETE /api/v1/privacy/{profile_id}
```

This removes the server-side baseline, longitudinal history and personalization records associated with the profile ID.

## HTTP hardening

The backend includes configurable:

- request IDs
- request-body size limits
- security headers
- no-store API responses
- rate limiting
- optional API-key authentication
- optional HSTS
- restrictive CORS configuration

## WebSocket hardening

Live sockets support:

- browser origin checking
- concurrent-connection limits
- optional token authentication
- existing audio/chunk/session limits

## Deployment limitation

The built-in rate limiter and connection counter are process-local. Public multi-worker deployments should put CEREBRO behind a reverse proxy/gateway with TLS termination, centralized authentication where appropriate, and shared rate limiting.

See [`docs/SECURITY.md`](docs/SECURITY.md) for the detailed threat model and hardening guidance.

---

# 20. Repository structure

```text
CEREBRO/
│
├── README.md
├── .gitignore
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   ├── EVALUATION.md
│   ├── SECURITY.md
│   └── SETUP.md
│
├── backend/
│   ├── app/
│   │   ├── affect_dynamics/
│   │   ├── analyzers/
│   │   ├── attribution/
│   │   ├── audio/
│   │   ├── comparison/
│   │   ├── conversation_graph/
│   │   ├── conversation_state/
│   │   ├── cross_session_patterns/
│   │   ├── datasets/
│   │   ├── deep_emotion/
│   │   ├── dimensional_emotion/
│   │   ├── drift_engine/
│   │   ├── early_warning/
│   │   ├── emotion_baseline/
│   │   ├── evaluation/
│   │   ├── explainability/
│   │   ├── features/
│   │   ├── fusion/
│   │   ├── graph_engine/
│   │   ├── live/
│   │   ├── live_multi_speaker/
│   │   ├── longitudinal_emotion/
│   │   ├── models/
│   │   ├── optimization/
│   │   ├── personal_baseline/
│   │   ├── personalization/
│   │   ├── privacy/
│   │   ├── quality/
│   │   ├── robustness/
│   │   ├── routers/
│   │   ├── services/
│   │   ├── speaker_diarization/
│   │   ├── temporal/
│   │   ├── transcription/
│   │   ├── utils/
│   │   ├── voice_change/
│   │   ├── config.py
│   │   ├── main.py
│   │   └── security.py
│   │
│   ├── scripts/
│   │   ├── inspect_ravdess.py
│   │   ├── train_baseline.py
│   │   ├── optimize_baseline.py
│   │   ├── generate_fusion_predictions.py
│   │   ├── train_fusion.py
│   │   └── run_example.py
│   │
│   ├── tests/
│   ├── data/
│   │   └── examples/
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/
    ├── public/
    ├── src/
    │   ├── audio/
    │   ├── components/
    │   ├── legacy/
    │   ├── App.jsx
    │   ├── api.js
    │   ├── main.jsx
    │   └── styles.css
    ├── package.json
    ├── package-lock.json
    ├── vite.config.js
    └── .env.example
```

The `legacy/` frontend folder contains the isolated predecessor text-thread interface retained for backwards compatibility with the original ToxiGraph lineage. It is not part of the active CEREBRO page composition.

---

# 21. Local setup

## Prerequisites

Recommended:

- Python 3.11+
- Node.js 20+
- Git
- a modern browser with microphone support
- FFmpeg / current audio backend requirements needed by the installed pyannote/torchcodec stack
- a Hugging Face account/token for models requiring gated or authenticated access

## Backend

```bash
cd backend
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --port 8000
```

### macOS/Linux

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL printed in the terminal and allow microphone access.

For deployed/non-local microphone access, use HTTPS and configure the exact frontend origin in `ALLOWED_ORIGINS`.

---

# 22. Environment configuration

The backend template is intentionally explicit. Important variables include:

```env
APP_NAME=CEREBRO
APP_VERSION=0.27.0
API_PREFIX=/api/v1
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000
MODEL_MODE=auto

CLASSICAL_MODEL_PATH=models/cerebro_classical_emotion.joblib
DEEP_EMOTION_MODEL=Dpngtm/wav2vec2-emotion-recognition
FUSION_CALIBRATION_PATH=models/cerebro_fusion_calibration.joblib
ASR_MODEL=openai/whisper-small
DIMENSIONAL_EMOTION_ENABLED=true
DIMENSIONAL_EMOTION_MODEL=3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes

PYANNOTE_TOKEN=
SPEAKER_DIARIZATION_ENABLED=true
SPEAKER_DIARIZATION_MODEL=pyannote/speaker-diarization-community-1

LIVE_MULTI_SPEAKER_ENABLED=true
LIVE_MULTI_SPEAKER_CONTEXT_SECONDS=8.0
LIVE_MULTI_SPEAKER_HOP_SECONDS=1.0
LIVE_MULTI_SPEAKER_STABILIZATION_SECONDS=0.8
LIVE_MULTI_SPEAKER_MAX_DURATION_SECONDS=180

SECURITY_HEADERS_ENABLED=true
SECURITY_MAX_BODY_BYTES=20971520
HTTP_RATE_LIMIT_ENABLED=true
HTTP_RATE_LIMIT_REQUESTS=120
HTTP_RATE_LIMIT_WINDOW_SECONDS=60
API_AUTH_ENABLED=false
API_AUTH_KEY=
SECURITY_HSTS_ENABLED=false
WEBSOCKET_AUTH_ENABLED=false
WEBSOCKET_AUTH_TOKEN=
WEBSOCKET_MAX_CONNECTIONS_PER_IP=3

PRIVACY_RETENTION_ENABLED=true
PRIVACY_RETENTION_DAYS=30
```

## Production recommendation

For a public deployment:

```env
ALLOWED_ORIGINS=https://your-frontend.example
API_AUTH_ENABLED=true
API_AUTH_KEY=<long-random-secret>
WEBSOCKET_AUTH_ENABLED=true
WEBSOCKET_AUTH_TOKEN=<long-random-secret>
SECURITY_HSTS_ENABLED=true
```

Use a reverse proxy for TLS, centralized rate limiting and infrastructure-level logging/monitoring.

Never commit `.env` files or secrets.

---

# 23. API surface

The API is organized by capability rather than by UI component.

| Capability | Representative endpoint |
|---|---|
| Health | `GET /api/v1/health/` |
| Audio validation | `POST /api/v1/audio/validate` |
| Acoustic features | `POST /api/v1/audio/features` |
| Classical emotion | `POST /api/v1/audio/emotion` |
| Deep emotion | `POST /api/v1/audio/emotion/deep` |
| Fusion | `POST /api/v1/audio/emotion/fusion` |
| Insights | `POST /api/v1/audio/emotion/insights` |
| Temporal emotion | `POST /api/v1/audio/emotion/temporal` |
| Continuous affect | `POST /api/v1/audio/emotion/affect` |
| Affect dynamics | `POST /api/v1/audio/emotion/affect/dynamics` |
| Recording comparison | `POST /api/v1/audio/emotion/compare` |
| Transcript | `POST /api/v1/audio/transcribe` |
| Speaker-aware analysis | `POST /api/v1/audio/speakers` |
| Conversation interaction | returned through speaker-aware analysis |
| Live single-speaker | `WS /api/v1/audio/live/ws` |
| Live multi-speaker | `WS /api/v1/audio/live-multi/ws` |
| Personal baseline | `/api/v1/audio/baseline/*` |
| Longitudinal history | `/api/v1/audio/history/*` |
| Voice change | `/api/v1/audio/voice-change/*` |
| Cross-session patterns | `/api/v1/audio/history-patterns/*` |
| Personalized calibration | `/api/v1/audio/personalization/*` |
| Evaluation | `/api/v1/evaluation/*` |
| Optimization | `/api/v1/optimization/*` |
| Robustness | `/api/v1/robustness/*` |
| Privacy | `/api/v1/privacy/*` |

Each public capability also exposes a status endpoint when model/artifact availability can vary by environment.

---

# 24. Testing

The project follows a phase-gated engineering workflow:

```text
Implement
   ↓
Unit tests
   ↓
Integration tests
   ↓
Environment/manual verification
   ↓
Regression test
   ↓
Freeze
```

The current release passes:

```text
271 backend tests
271 passed
0 failed
```

The suite covers:

- audio validation and normalization
- acoustic features
- classical ML contracts
- deep-model adapters
- fusion/calibration
- temporal inference
- continuous affect
- affect dynamics
- uncertainty/explainability
- personal baselines
- recording comparison
- ASR/content verification
- audio quality
- robustness
- evaluation infrastructure
- speaker diarization
- interaction graphs
- conversation dynamics
- conversation states
- live streaming
- longitudinal storage
- anomaly detection
- cross-session patterns
- explicit-feedback personalization
- privacy controls
- HTTP security
- WebSocket security

Run the complete backend suite:

```bash
cd backend
pytest -q
```

---

# 25. Release and submission hygiene

This repository is intentionally source-only.

Before committing:

```text
[✓] no .env files
[✓] no API keys/tokens
[✓] no raw audio recordings
[✓] no downloaded datasets
[✓] no pretrained model weights
[✓] no generated personal profile data
[✓] no Python bytecode caches
[✓] no pytest caches
[✓] no frontend node_modules
[✓] no Vite dist output
[✓] no repetitive phase acceptance clutter
```

The project keeps only durable technical documentation under `docs/`:

```text
docs/
├── ARCHITECTURE.md
├── DEVELOPMENT.md
├── EVALUATION.md
├── SECURITY.md
└── SETUP.md
```

---

# 26. Limitations and responsible interpretation

## Emotion recognition is expressed-affect inference

CEREBRO predicts patterns associated with emotion **conveyed through speech**. It does not establish a person's true internal emotional state.

## Model bias and domain shift

Speech-emotion models can perform differently across speakers, languages, microphones, recording conditions, cultures, and acting styles. Reported upstream metrics should not be assumed to transfer to every environment.

## Attribution is not causality

A counterfactual contribution score can identify a candidate input that influenced a detected transition under the model, but it is not proof of psychological causation.

## Live diarization is rolling-context

The live multi-speaker mode uses bounded re-diarization rather than a fully causal streaming diarizer. Recent speaker assignments can therefore be revised as more context arrives.

## Personalization requires evidence

CEREBRO uses minimum sample/evidence gates before activating personal baselines, cross-session patterns, anomaly labels, or speaker-specific calibration.

## Privacy controls are application-level

The repository contains useful hardening, but a public production deployment still needs TLS, infrastructure isolation, shared rate limiting, secret management, monitoring, dependency scanning, and appropriate identity/access management.

---

# 27. Why CEREBRO is more than an emotion classifier

The complete system can be summarized as:

```text
SPEECH
  ↓
Audio quality
  ↓
Acoustic representation
  +
Deep speech representation
  ↓
Calibrated model fusion
  ↓
Emotion
  +
Valence / Arousal / Dominance
  +
Uncertainty
  ↓
Temporal affect dynamics
  ↓
Personal baseline
  ↓
Longitudinal history
  ↓
Anomaly + recurring pattern discovery
  ↓
Explicit-feedback personalization
  ↓
Speaker-aware analysis
  ↓
Conversation interaction graph
  ↓
Conversation dynamics
  ↓
Conversation state
  ↓
Real-time multi-speaker intelligence
  ↓
Privacy + security controls
```

That is the central idea behind CEREBRO: **treat speech emotion as a layered inference problem rather than a one-shot label.**

---

# 28. Final submission note

CEREBRO is submitted as a serious end-to-end engineering and ML project for **ALGOTHON'26**.

The repository is structured so a reviewer can:

1. inspect the architecture,
2. install the backend/frontend,
3. run the automated suite,
4. supply the required model/dataset assets,
5. exercise the single-speaker pipeline,
6. inspect temporal affect and explanations,
7. use the Comparison Lab,
8. test speaker-aware and live multi-speaker workflows,
9. inspect evaluation/robustness tooling, and
10. review privacy/security behavior.

CEREBRO intentionally favors **transparent evidence, reproducible evaluation, explicit uncertainty, and responsible interpretation** over unsupported claims.

---

## Built with

**Python · FastAPI · NumPy · scikit-learn · PyTorch · Hugging Face Transformers · pyannote.audio · React 19 · Vite · Web Audio API · WebSockets**

---

## Project version

```text
CEREBRO v0.27.0
ALGOTHON'26 submission build
```
