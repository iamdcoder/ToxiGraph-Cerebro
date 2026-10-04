import { useEffect, useMemo, useState } from "react";

import { analyzeCurrentVoiceChange } from "../api";

function pct(value) {
  return `${(value * 100).toFixed(0)}%`;
}

function formatUnit(value, unit) {
  if (unit === "Hz") return `${value.toFixed(0)} Hz`;
  if (unit === "RMS") return value.toFixed(4);
  if (unit === "ratio") return value.toFixed(2);
  if (unit === "syllables/s") return `${value.toFixed(2)}/s`;
  return value.toFixed(2);
}

const LEVEL_LABEL = {
  insufficient_data: "INSUFFICIENT HISTORY",
  normal: "NORMAL RANGE",
  minor_change: "MINOR CHANGE",
  significant_change: "SIGNIFICANT CHANGE",
  unusual: "UNUSUAL",
};

export default function VoiceChangePanel({ profileId, currentSession }) {
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const payload = useMemo(() => currentSession, [currentSession]);

  useEffect(() => {
    if (!profileId || !payload) {
      setResult(null);
      return;
    }
    let mounted = true;
    setBusy(true);
    setError("");
    analyzeCurrentVoiceChange(profileId, payload)
      .then((response) => {
        if (mounted) setResult(response.report);
      })
      .catch((err) => {
        if (mounted) setError(err?.message || "Voice-change analysis is unavailable.");
      })
      .finally(() => {
        if (mounted) setBusy(false);
      });
    return () => {
      mounted = false;
    };
  }, [profileId, payload]);

  if (!profileId || !payload) return null;

  return (
    <section className="voice-change-panel panel-card">
      <div className="result-heading">
        <div>
          <p className="eyebrow">PHASE 22 · VOICE CHANGE & ANOMALY</p>
          <h3>Is this recording unusual for this voice?</h3>
          <p className="muted-copy">
            CEREBRO compares the current recording with prior saved sessions using robust historical ranges. This is change detection, not a psychological diagnosis.
          </p>
        </div>
        {busy ? (
          <span className="result-badge warning">ANALYZING</span>
        ) : result ? (
          <span className={`result-badge ${result.level === "normal" ? "success" : "warning"}`}>
            {LEVEL_LABEL[result.level] || result.level}
          </span>
        ) : null}
      </div>

      {error && <p className="baseline-error" role="alert">{error}</p>}

      {result && (
        <>
          <div className="voice-change-metrics">
            <div className="metric-item"><span>Change score</span><strong>{pct(result.change_score)}</strong></div>
            <div className="metric-item"><span>Detection confidence</span><strong>{pct(result.confidence)}</strong></div>
            <div className="metric-item"><span>Prior sessions</span><strong>{result.historical_session_count}</strong></div>
            <div className="metric-item"><span>Reference span</span><strong>{result.reference_time_span_days.toFixed(1)} d</strong></div>
          </div>

          <div className="voice-change-layout">
            <div className="voice-change-summary-card">
              <span className="eyebrow">INTERPRETATION</span>
              <p>{result.summary}</p>
              {result.quality_limited && <span className="model-note">Recording quality limits confidence in this comparison.</span>}
            </div>
            <div className="voice-change-evidence-card">
              <div className="feature-section-heading"><span>STRONGEST DEVIATIONS</span><small>vs. historical median</small></div>
              <div className="voice-change-list">
                {result.strongest_changes.map((item) => (
                  <div className="voice-change-row" key={`${item.category}-${item.name}`}>
                    <div>
                      <strong>{item.name}</strong>
                      <small>{item.direction} · {item.significance}</small>
                    </div>
                    <span>{item.robust_z.toFixed(2)}σ</span>
                    <small>{formatUnit(item.current_value, item.unit)} / {formatUnit(item.historical_median, item.unit)}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>
          <p className="model-note">{result.caveat}</p>
        </>
      )}
    </section>
  );
}
