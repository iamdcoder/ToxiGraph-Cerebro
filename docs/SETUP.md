# CEREBRO Setup

## Prerequisites

- Python 3.11+ recommended
- Node.js 20+ recommended
- Modern Chromium/Edge/Firefox/Safari with microphone support
- Git
- FFmpeg for the current pyannote/torchcodec audio stack when speaker diarization is enabled
- Hugging Face account/token for models that require accepted access conditions

## Backend

```bash
cd backend
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

# Windows
copy .env.example .env

# macOS/Linux
# cp .env.example .env

uvicorn app.main:app --reload --port 8000
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Use the Vite URL shown in the terminal. Allow microphone access. For non-local microphone use, serve the frontend over HTTPS.

## Environment

Important variables include:

```env
API_PREFIX=/api/v1
ALLOWED_ORIGINS=http://localhost:5173
CLASSICAL_MODEL_PATH=models/cerebro_classical_emotion.joblib
DEEP_EMOTION_MODEL=Dpngtm/wav2vec2-emotion-recognition
FUSION_CALIBRATION_PATH=models/cerebro_fusion_calibration.joblib
ASR_MODEL=openai/whisper-small
DIMENSIONAL_EMOTION_MODEL=3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes
PYANNOTE_TOKEN=
SPEAKER_DIARIZATION_MODEL=pyannote/speaker-diarization-community-1
LIVE_MULTI_SPEAKER_ENABLED=true
LIVE_MULTI_SPEAKER_CONTEXT_SECONDS=8.0
LIVE_MULTI_SPEAKER_HOP_SECONDS=1.0
LIVE_MULTI_SPEAKER_STABILIZATION_SECONDS=0.8
```

See `backend/app/config.py` for the full configuration surface.


## Security and privacy settings

Local development can use the defaults in `backend/.env.example`. For a public deployment, set the exact frontend origin and enable authentication controls such as:

```env
ALLOWED_ORIGINS=https://your-frontend.example
API_AUTH_ENABLED=true
API_AUTH_KEY=<long-random-secret>
WEBSOCKET_AUTH_ENABLED=true
WEBSOCKET_AUTH_TOKEN=<long-random-secret>
SECURITY_HSTS_ENABLED=true
```

Profile retention defaults to 30 days and can be configured with `PRIVACY_RETENTION_DAYS`. Use `DELETE /api/v1/privacy/{profile_id}` to remove all server-side profile data for one profile.

## Training assets

Do not commit datasets, model weights, generated recordings, or personal profile data. Paths for those artifacts are already covered by `.gitignore`.

## RAVDESS

Download/extract the dataset separately, then inspect it:

```bash
cd backend
python -m scripts.inspect_ravdess --data-dir PATH_TO_RAVDESS
```

## Hugging Face speaker diarization

For `pyannote/speaker-diarization-community-1`, accept the model conditions and provide a Hugging Face token. Follow the current model card and pyannote documentation for the exact dependency/runtime requirements.

## First-run verification

```bash
cd backend
pytest -q
```

Then exercise:

1. microphone recording
2. single-speaker live analysis
3. batch speaker-aware analysis
4. live multi-speaker analysis
5. personal history and comparison features

## Useful commands

See the main README for training, optimization, fusion, and benchmark commands.
