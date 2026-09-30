# ToxiGraph

> Trace the fracture point. Predict the next one.

ToxiGraph is a graph-aware conversation analysis prototype for PS-03 — Sentiment Drift in Social Threads.

## What the current build does

The application turns reply data into a normalized thread DAG, runs per-comment sentiment, toxicity, emotion and linguistic analysis, computes structural graph features, detects emotional drift with Bayesian online change-point logic, estimates a strongest trigger with counterfactual contribution, and produces an explainable precursor-risk forecast.

The frontend presents the result as an interactive branch graph, sentiment/forecast timeline, risk panel, attribution panel, comment inspector and forensic report.

## Stack

Backend: Python, FastAPI, Pydantic, Hugging Face Transformers.

Frontend: React, Vite, custom SVG visualization.

No D3 runtime dependency is required by the current frontend.

## Run locally

### Backend

```bash
cd backend
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env   # Windows
# cp .env.example .env  # macOS/Linux

uvicorn app.main:app --reload --port 8000
```

### Frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal.

## Model reliability

`MODEL_MODE=auto` is the recommended demo setting. The analyzers first try the configured Hugging Face models. If the package, model cache, download or inference fails, the application automatically falls back to deterministic local heuristics so the dashboard remains usable.

Use `MODEL_MODE=heuristic` to force the offline fallback for a fully deterministic demo.

The fallback is a resilience path, not a substitute for the pretrained models. Judge-facing claims should distinguish the two modes.

## Demo thread

The checked-in demo contains 36 comments across multiple branches and is deliberately shaped as:

`constructive → disagreement → personal attribution → toxic escalation → cool-down`

The same fixture is kept in both:

- `backend/data/demo/demo_thread.json`
- `frontend/public/demo_thread.json`

Run `MODEL_MODE=heuristic` for the most reproducible local demo.

## Tests

```bash
cd backend
pytest -q
```

The current backend test suite passes with 17 tests in the deadline build.

## Demo flow

1. Start backend.
2. Start frontend.
3. Load the demo thread.
4. Show the wider multi-branch graph.
5. Select the fracture marker.
6. Select the trigger candidate.
7. Move to the timeline and show the fracture line.
8. Show that the forecast rises before the realized toxicity event.
9. Show counterfactual attribution evidence.
10. Open the forensic report.

## Deadline scope

The implementation plan identifies graph construction, sentiment analysis, timeline visualization and BOCPD as P0, with toxicity/reporting/annotation as P1 and GAT/early-warning plus Granger/SHAP as later priorities. The current build therefore favors a reliable end-to-end demo over adding a last-minute research model that could destabilize the system.
