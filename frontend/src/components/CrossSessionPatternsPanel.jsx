import { useEffect, useState } from "react";

import { getCrossSessionPatterns } from "../api";

function pct(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function titleCase(value) {
  return String(value || "").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function patternLabel(type) {
  return String(type || "").replaceAll("_", " ");
}

export default function CrossSessionPatternsPanel({ profileId, refreshKey = 0 }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function refresh() {
    if (!profileId) return;
    setLoading(true);
    setError("");
    try {
      const response = await getCrossSessionPatterns(profileId);
      setResult(response.report);
    } catch (err) {
      setError(err?.message || "Cross-session patterns could not be loaded.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    if (!profileId) {
      setResult(null);
      return;
    }
    refresh();
  }, [profileId, refreshKey]);

  if (!profileId) return null;

  return (
    <section className="cross-session-patterns-panel">
      <div className="result-heading">
        <div>
          <p className="eyebrow">PHASE 23 · CROSS-SESSION PATTERN ENGINE</p>
          <h3>Find recurring patterns across this voice history</h3>
          <p className="muted-copy">
            CEREBRO compares saved session summaries to identify recurring emotion sequences, affect direction patterns, stability/volatility, and descriptive acoustic–affect associations.
          </p>
        </div>
        <span className={`result-badge ${result?.available ? "success" : "warning"}`}>
          {loading ? "SEARCHING" : result?.available ? "PATTERNS ONLINE" : "NEEDS HISTORY"}
        </span>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {!result?.available ? (
        <div className="pattern-empty-state">
          <strong>{result?.summary || "Save at least four analyzed sessions to search for recurring patterns."}</strong>
          <span>{result?.caveat || "The engine intentionally avoids making pattern claims from very small histories."}</span>
        </div>
      ) : (
        <>
          <div className="pattern-metric-grid">
            <div className="metric-item"><span>Pattern count</span><strong>{result.patterns.length}</strong></div>
            <div className="metric-item"><span>Recurrence index</span><strong>{pct(result.recurrence_index)}</strong></div>
            <div className="metric-item"><span>Labeled sessions</span><strong>{result.labeled_session_count}</strong></div>
            <div className="metric-item"><span>Affect transitions</span><strong>{result.usable_affect_transitions}</strong></div>
            <div className="metric-item"><span>Dominant emotion</span><strong>{titleCase(result.dominant_emotion) || "—"}</strong></div>
            <div className="metric-item"><span>Dominant share</span><strong>{result.dominant_emotion_share == null ? "—" : pct(result.dominant_emotion_share)}</strong></div>
          </div>

          <div className="pattern-grid">
            <div className="cross-pattern-card">
              <div className="feature-section-heading"><span>DISCOVERED PATTERNS</span><small>{result.patterns.length} signals</small></div>
              <div className="pattern-list">
                {result.patterns.length === 0 ? (
                  <p className="muted-copy">No strong recurring pattern met the current evidence thresholds.</p>
                ) : result.patterns.map((pattern, index) => (
                  <article className="pattern-row" key={`${pattern.pattern_type}-${pattern.title}-${index}`}>
                    <div className="pattern-row-top">
                      <div>
                        <small>{patternLabel(pattern.pattern_type)}</small>
                        <strong>{pattern.title}</strong>
                      </div>
                      <span>{pct(pattern.strength)}</span>
                    </div>
                    <p>{pattern.description}</p>
                    <div className="pattern-bar"><i style={{ width: `${Math.max(4, pattern.strength * 100)}%` }} /></div>
                    <div className="pattern-row-meta">
                      <span>{pattern.occurrences} occurrences</span>
                      <span>confidence {pct(pattern.confidence)}</span>
                    </div>
                  </article>
                ))}
              </div>
            </div>

            <div className="cross-pattern-card">
              <div className="feature-section-heading"><span>REPEATED EMOTION TRANSITIONS</span><small>consecutive sessions</small></div>
              {result.repeated_transitions.length === 0 ? (
                <p className="muted-copy">No transition repeated strongly enough to be surfaced.</p>
              ) : (
                <div className="transition-pattern-list">
                  {result.repeated_transitions.map((item) => (
                    <div className="transition-pattern-row" key={`${item.from_emotion}-${item.to_emotion}`}>
                      <strong>{titleCase(item.from_emotion)} → {titleCase(item.to_emotion)}</strong>
                      <span>{item.occurrences}×</span>
                      <small>{pct(item.share_of_transitions)} of transitions</small>
                    </div>
                  ))}
                </div>
              )}
              <div className="pattern-distribution-card">
                <span>EMOTION DISTRIBUTION ENTROPY</span>
                <strong>{result.emotion_entropy == null ? "—" : pct(result.emotion_entropy)}</strong>
                <small>Lower values mean the saved sessions are concentrated in fewer labeled emotion categories.</small>
              </div>
            </div>
          </div>

          <p className="pattern-summary">{result.summary}</p>
          <p className="model-note">{result.caveat}</p>
        </>
      )}
    </section>
  );
}
