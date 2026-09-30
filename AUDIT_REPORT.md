# ToxiGraph — Current Project Audit

## Critical blockers found in the uploaded build

### 1. Frontend parser failure

`frontend/src/components/ThreadGraph.jsx` contained two adjacent `branchGap` tokens in the `leafWidth` calculation. That is invalid JavaScript/JSX syntax and prevents Vite from parsing the component.

**Fix:** replace the entire component with the deadline-safe graph layout in this build.

### 2. Hero graph was visually constrained

The SVG used a hardcoded `1600`-pixel canvas, a fixed `820px` height and `preserveAspectRatio="xMidYMin meet"`. A large thread therefore got scaled down instead of becoming a large, scrollable graph.

**Fix:** compute width from the number of leaves and give the SVG its real pixel width/height. The surrounding `.graph-scroll` container provides horizontal scrolling.

### 3. Demo duplication meant the bigger demo could still be ignored

The frontend loads `/demo_thread.json`, which comes from `frontend/public/demo_thread.json`. Updating only `backend/data/demo/demo_thread.json` would not change what judges see in the UI.

**Fix:** keep both fixture files identical and add a test that fails if they diverge.

### 4. Model dependency failure could take down analysis

The original analyzers imported `transformers`/`torch` at module import time and immediately loaded three models. A missing package, uncached model or inference error could turn a demo into a 500 response.

**Fix:** analyzers now load models defensively and use a deterministic local fallback in `auto` mode. This preserves the transformer path when available while protecting the demo path.

### 5. Pydantic validation errors were displayed poorly in the frontend

FastAPI validation errors can return `detail` as an array of structured objects. The original frontend passed that array directly to `new Error`, producing unreadable `[object Object]` messages.

**Fix:** `frontend/src/api.js` converts structured validation errors into a readable one-line message.

### 6. The timeline had the same scaling problem as the graph

The timeline used a fixed `980`-pixel SVG, so longer threads compressed into the same width.

**Fix:** timeline width now grows with comment count and remains horizontally scrollable.

### 7. BOCPD short-run probability reported a misleading early value

The recent-change probability is defined as posterior mass over short run lengths. During the first few observations, every possible run length falls inside that short window, so a literal calculation can produce `1.0` without enough history for a useful online warning.

**Fix:** suppress that signal until the detector has more observations than its short window. The exact run-length-zero probability remains available separately.

## Important non-blockers

- Current graph rendering is a custom SVG tree, not a D3 force-directed graph.
- Current early warning is an explainable precursor-risk heuristic, not a trained LightGBM model.
- Current attribution is counterfactual contribution analysis; it should not be described as proof of causality.
- GAT, Granger and SHAP are not implemented in the current build.
- Live Reddit/YouTube/X ingestion is not part of the deadline-safe path.
- CSV ingestion is not yet the core demo path.

These are scope decisions, not reasons to destabilize the demo late in the build.

## Verified state

- Backend syntax compilation: passed.
- Backend tests: 18 passed.
- Offline/auto analyzer fallback: verified.
- 36-comment end-to-end demo analysis: HTTP 200.
- Demo fracture: `comment_21` in heuristic mode.
- Strongest attribution trigger: `comment_16` in heuristic mode.
- Realized toxicity event: `comment_26` (0-based realized index 25) in heuristic mode.
- Earliest actionable warning: `comment_23`, three comments before the realized event.

## Submission hygiene

Do not ship `backend/.venv`, `frontend/node_modules` or generated cache directories in the submission ZIP. They are platform-specific, huge and can contain broken native binaries when moved between operating systems.
