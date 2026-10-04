import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  analyzeThread,
  loadExampleThread,
} from "./api";

import ThreadGraph from "./components/ThreadGraph";
import SentimentTimeline from "./components/SentimentTimeline";
import RiskPanel from "./components/RiskPanel";
import FracturePanel from "./components/FracturePanel";
import AttributionPanel from "./components/AttributionPanel";
import CommentInspector from "./components/CommentInspector";
import ForensicReport from "./components/ForensicReport";


function formatDuration(
  nodes
) {
  if (!nodes?.length) {
    return "—";
  }

  const ordered = [
    ...nodes,
  ].sort(
    (a, b) =>
      new Date(a.timestamp) -
      new Date(b.timestamp)
  );

  const start = new Date(
    ordered[0].timestamp
  );

  const end = new Date(
    ordered[
      ordered.length - 1
    ].timestamp
  );

  const minutes =
    Math.max(
      0,
      Math.round(
        (end - start) /
          60000
      )
    );

  if (minutes < 60) {
    return `${minutes}m`;
  }

  return `${Math.floor(
    minutes / 60
  )}h ${minutes % 60}m`;
}


function riskClass(
  risk
) {
  if (risk >= 0.75) {
    return "critical";
  }

  if (risk >= 0.55) {
    return "warning";
  }

  if (risk >= 0.30) {
    return "watch";
  }

  return "safe";
}


function countBranches(
  nodes
) {
  if (!nodes?.length) {
    return 0;
  }

  const roots =
    nodes
      .filter(
        (node) =>
          node.parent_id === null
      )
      .map(
        (node) =>
          node.id
      );

  const rootSet =
    new Set(roots);

  const branches =
    new Set(
      nodes
        .map(
          (node) =>
            node
              ?.features
              ?.branch_root_id
        )
        .filter(
          (branchId) =>
            branchId &&
            !rootSet.has(
              branchId
            )
        )
    );

  if (branches.size > 0) {
    return branches.size;
  }

  return roots.length;
}


export default function App() {
  const [thread, setThread] =
    useState(null);

  const [selectedId, setSelectedId] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [loadingExample, setLoadingExample] =
    useState(false);

  const [error, setError] =
    useState("");

  const [analysisMeta, setAnalysisMeta] =
    useState(null);

  const fileInputRef =
    useRef(null);


  const selectedNode =
    useMemo(
      () =>
        thread?.nodes?.find(
          (node) =>
            node.id ===
            selectedId
        ) || null,
      [
        thread,
        selectedId,
      ]
    );


  const risk =
    Number(
      thread
        ?.early_warning
        ?.current_risk ||
      0
    );


  const loadAndAnalyze =
    useCallback(
      async (
        rawThread
      ) => {
        setLoading(true);
        setError("");

        try {
          const result =
            await analyzeThread(
              rawThread
            );

          const analyzed =
            result
              ?.analyzed_thread;

          if (!analyzed) {
            throw new Error(
              "Backend returned no analyzed thread."
            );
          }

          setThread(
            analyzed
          );

          setAnalysisMeta(
            result?.metadata || null
          );

          const fractureId =
            analyzed
              ?.drift_analysis
              ?.global_timeline
              ?.fracture_point
              ?.comment_id;

          setSelectedId(
            fractureId ||
              analyzed
                ?.attribution
                ?.primary_trigger_id ||
              analyzed
                ?.nodes?.[0]
                ?.id ||
              null
          );

        } catch (err) {
          setError(
            err?.message ||
              "Unable to analyze thread."
          );
        } finally {
          setLoading(false);
        }
      },
      []
    );


  const handleLoadExample =
    useCallback(
      async () => {
        setLoadingExample(true);
        setError("");

        try {
          const example =
            await loadExampleThread();

          await loadAndAnalyze(
            example
          );

        } catch (err) {

          setError(
            err?.message ||
              "Unable to load example thread."
          );

        } finally {
          setLoadingExample(false);
        }
      },
      [loadAndAnalyze]
    );


  const handleFile =
    async (
      event
    ) => {
      const file =
        event.target.files?.[0];

      if (!file) {
        return;
      }

      setError("");

      try {

        if (
          !file.name
            .toLowerCase()
            .endsWith(".json")
        ) {
          throw new Error(
            "For this build, upload a JSON thread export."
          );
        }

        const text =
          await file.text();

        const parsed =
          JSON.parse(text);

        if (
          !parsed.nodes ||
          !Array.isArray(
            parsed.nodes
          )
        ) {
          throw new Error(
            "The JSON must contain a nodes array."
          );
        }

        await loadAndAnalyze(
          parsed
        );

      } catch (err) {

        setError(
          err?.message ||
            "Invalid JSON thread."
        );

      } finally {
        event.target.value = "";
      }
    };


  useEffect(() => {
    handleLoadExample();
  }, [handleLoadExample]);


  const comments =
    thread?.nodes?.length ||
    0;

  const branchCount =
    countBranches(
      thread?.nodes
    );

  const currentRiskLabel =
    thread
      ?.early_warning
      ?.current_risk_level ||
    "safe";


  return (
    <div className="app-shell">

      <header className="topbar">

        <div className="brand">

          <div className="brand-mark">
            <span />
            <span />
            <span />
          </div>

          <div>
            <div className="brand-name">
              TOXIGRAPH
            </div>

            <div className="brand-subtitle">
              SENTIMENT DRIFT INTELLIGENCE
            </div>
          </div>

        </div>


        <div className="topbar-actions">

          <div
            className={`live-indicator ${
              analysisMeta?.overall_mode || "pending"
            }`}
            title={
              analysisMeta
                ? `Sentiment: ${analysisMeta.sentiment_mode} · Toxicity: ${analysisMeta.toxicity_mode} · Emotion: ${analysisMeta.emotion_mode}`
                : "Analysis engine status"
            }
          >
            <span className="live-dot" />
            <span>
              {analysisMeta?.overall_mode === "transformer"
                ? "TRANSFORMER ENGINE"
                : analysisMeta?.overall_mode === "hybrid"
                ? "HYBRID ENGINE"
                : analysisMeta?.overall_mode === "heuristic"
                ? "OFFLINE ENGINE"
                : "ANALYSIS ENGINE"}
            </span>
          </div>

          <button
            className="secondary-button"
            onClick={
              handleLoadExample
            }
            disabled={
              loading ||
              loadingExample
            }
          >
            {loadingExample
              ? "Analyzing…"
              : "Load example"}
          </button>

          <button
            className="primary-button"
            onClick={() =>
              fileInputRef.current?.click()
            }
            disabled={
              loading
            }
          >
            Analyze thread
          </button>

          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            onChange={
              handleFile
            }
            hidden
          />

        </div>

      </header>


      <main className="dashboard">

        <section className="hero-row">

          <div className="hero-copy">

            <span className="eyebrow">
              CEREBRO · PS-03
            </span>

            <h1>
              Trace the fracture point.
              <br />
              <span>
                Predict the next one.
              </span>
            </h1>

            <p>
              ToxiGraph reconstructs how a
              conversation changes over time,
              identifies the reply associated with
              the strongest emotional fracture, and
              estimates whether escalation was
              foreseeable.
            </p>

          </div>


          <div className="hero-stats">

            <div>
              <span>
                COMMENTS
              </span>

              <strong>
                {comments}
              </strong>
            </div>

            <div>
              <span>
                BRANCHES
              </span>

              <strong>
                {branchCount}
              </strong>
            </div>

            <div>
              <span>
                THREAD DURATION
              </span>

              <strong>
                {formatDuration(
                  thread?.nodes
                )}
              </strong>
            </div>

          </div>

        </section>


        {error && (
          <div className="error-banner">

            <strong>
              Analysis error
            </strong>

            <span>
              {error}
            </span>

            <button
              onClick={() =>
                setError("")
              }
            >
              ×
            </button>

          </div>
        )}


        {loading && (
          <div className="loading-banner">

            <div className="loading-spinner" />

            <div>
              <strong>
                ToxiGraph is reading the thread
              </strong>

              <span>
                Running semantic, structural,
                drift, attribution and
                early-warning analysis…
              </span>
            </div>

          </div>
        )}


        {thread && (
          <>

            <section className="top-grid">

              <RiskPanel
                earlyWarning={
                  thread.early_warning
                }
              />

              <FracturePanel
                thread={thread}
                onSelect={
                  setSelectedId
                }
              />

            </section>


            <section className="panel graph-panel">

              <div className="panel-heading">

                <div>
                  <span className="eyebrow">
                    THREAD TOPOLOGY
                  </span>

                  <h2>
                    Conversation graph
                  </h2>
                </div>


                <div className="graph-heading-meta">

                  <span>
                    {thread.platform}
                  </span>

                  <span>
                    {comments} nodes
                  </span>

                  <span>
                    {branchCount} branches
                  </span>

                  <span
                    className={`mini-risk ${riskClass(
                      risk
                    )}`}
                  >
                    {currentRiskLabel}
                  </span>

                  {analysisMeta?.fallback_active && (
                    <span className="engine-note">
                      offline fallback
                    </span>
                  )}

                </div>

              </div>


              <div className="graph-explainer">

                <span>
                  Node color = sentiment
                </span>

                <span>
                  Node size = structural / drift impact
                </span>

                <span>
                  Edge weight = local influence
                </span>

                <span>
                  Ring = fracture / trigger
                </span>

              </div>


              <ThreadGraph
                thread={thread}
                selectedId={
                  selectedId
                }
                onSelect={
                  setSelectedId
                }
              />

            </section>


            <section className="panel timeline-panel">

              <div className="panel-heading">

                <div>
                  <span className="eyebrow">
                    TEMPORAL ANALYSIS
                  </span>

                  <h2>
                    Emotional arc
                  </h2>
                </div>

                <span className="timeline-note">
                  Bayesian drift + forecast
                </span>

              </div>


              <SentimentTimeline
                thread={thread}
              />

            </section>


            <section className="analysis-grid">

              <AttributionPanel
                thread={thread}
                onSelect={
                  setSelectedId
                }
              />


              <CommentInspector
                node={
                  selectedNode
                }
                thread={
                  thread
                }
                onClose={() =>
                  setSelectedId(
                    null
                  )
                }
              />

            </section>


            <ForensicReport
              thread={thread}
            />


            <section className="method-strip">

              <div>
                <span className="method-index">
                  01
                </span>

                <div>
                  <strong>
                    MULTI-SIGNAL
                  </strong>

                  <span>
                    sentiment · toxicity · emotion
                  </span>
                </div>
              </div>


              <div>
                <span className="method-index">
                  02
                </span>

                <div>
                  <strong>
                    BAYESIAN DRIFT
                  </strong>

                  <span>
                    online change-point probability
                  </span>
                </div>
              </div>


              <div>
                <span className="method-index">
                  03
                </span>

                <div>
                  <strong>
                    COUNTERFACTUAL
                  </strong>

                  <span>
                    candidate contribution testing
                  </span>
                </div>
              </div>


              <div>
                <span className="method-index">
                  04
                </span>

                <div>
                  <strong>
                    EARLY WARNING
                  </strong>

                  <span>
                    precursor-based risk forecast
                  </span>
                </div>
              </div>

            </section>

          </>
        )}

      </main>


      <footer className="footer">

        <span>
          TOXIGRAPH
        </span>

        <span>
          CEREBRO — NEURAL SENTIMENT HACKATHON
        </span>

        <span>
          PS-03 · SENTIMENT DRIFT
        </span>

      </footer>

    </div>
  );
}