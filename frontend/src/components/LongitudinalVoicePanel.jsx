import { useEffect, useMemo, useState } from "react";

import { deleteVoiceHistory, getVoiceHistory, saveVoiceHistorySession } from "../api";
import VoiceChangePanel from "./VoiceChangePanel";
import CrossSessionPatternsPanel from "./CrossSessionPatternsPanel";

const EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"];

function formatDate(value) {
  try {
    return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
  } catch {
    return value;
  }
}

function formatDelta(value) {
  if (value === null || value === undefined) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}

function payloadFromAnalysis({ recording, fusion, features, affect, insights }) {
  const uncertainty = insights?.insights?.uncertainty;
  const personal = insights?.insights?.personal_baseline;
  return {
    session_id: recording.sessionId || undefined,
    duration_seconds: recording.durationSeconds,
    emotion: fusion?.emotion || null,
    confidence: typeof fusion?.confidence === "number" ? fusion.confidence : null,
    quality_score: typeof uncertainty?.quality_score === "number" ? uncertainty.quality_score : null,
    baseline_deviation: typeof personal?.overall_deviation === "number" ? personal.overall_deviation : null,
    acoustic: {
      pitch_mean_hz: features?.pitch_hz?.mean ?? null,
      pitch_std_hz: features?.pitch_hz?.std ?? null,
      energy_mean: features?.energy?.mean ?? null,
      energy_std: features?.energy?.std ?? null,
      estimated_syllable_rate_sps: features?.estimated_syllable_rate_sps ?? null,
      estimated_pause_ratio: features?.estimated_pause_ratio ?? null,
      spectral_centroid_mean_hz: features?.spectral_centroid_hz?.mean ?? null,
    },
    affect: {
      valence: affect?.values?.valence ?? null,
      arousal: affect?.values?.arousal ?? null,
      dominance: affect?.values?.dominance ?? null,
    },
  };
}

export default function LongitudinalVoicePanel({ profileId, recording, fusion, features, affect, insights }) {
  const [history, setHistory] = useState(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [savedSessionKey, setSavedSessionKey] = useState(null);
  const [patternRefreshKey, setPatternRefreshKey] = useState(0);

  const canSave = Boolean(profileId && recording?.blob && recording?.durationSeconds > 0 && features?.features);

  async function refresh() {
    if (!profileId) return;
    try {
      const payload = await getVoiceHistory(profileId);
      setHistory(payload);
    } catch (err) {
      setError(err?.message || "Voice history could not be loaded.");
    }
  }

  useEffect(() => {
    setSavedSessionKey(null);
  }, [recording?.url]);

  useEffect(() => {
    let mounted = true;
    if (!profileId) {
      setHistory(null);
      return () => { mounted = false; };
    }
    getVoiceHistory(profileId)
      .then((payload) => {
        if (mounted) setHistory(payload);
      })
      .catch((err) => {
        if (mounted) setError(err?.message || "Voice history could not be loaded.");
      });
    return () => {
      mounted = false;
    };
  }, [profileId]);

  const latestSession = history?.sessions?.[history.sessions.length - 1] || null;
  const trendCards = useMemo(() => (
    (history?.trends?.affect_trends || []).map((item) => item)
  ), [history]);

  async function saveCurrentSession() {
    const sessionKey = recording?.url || recording?.blob?.size;
    if (!canSave || !sessionKey || savedSessionKey === sessionKey) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const payload = await saveVoiceHistorySession(
        profileId,
        payloadFromAnalysis({ recording, fusion, features: features.features, affect, insights }),
      );
      setHistory((current) => ({
        available: true,
        profile_id: profileId,
        session_count: payload.trends.session_count,
        max_sessions: 100,
        sessions: [
          ...(current?.sessions || []),
          payload.session,
        ].slice(-100),
        trends: payload.trends,
        message: "Voice history is active.",
      }));
      setSavedSessionKey(sessionKey);
      setPatternRefreshKey((value) => value + 1);
      setMessage("This analysis summary was added to your longitudinal voice history. Raw audio was not stored.");
    } catch (err) {
      setError(err?.message || "The analysis could not be saved.");
    } finally {
      setBusy(false);
    }
  }

  async function clearHistory() {
    if (!profileId || busy) return;
    setBusy(true);
    setError("");
    try {
      await deleteVoiceHistory(profileId);
      setHistory(null);
      setSavedSessionKey(null);
      setMessage("Longitudinal voice history deleted.");
    } catch (err) {
      setError(err?.message || "Voice history could not be deleted.");
    } finally {
      setBusy(false);
    }
  }

  const changeSessionPayload = useMemo(
    () => (canSave ? payloadFromAnalysis({ recording, fusion, features: features.features, affect, insights }) : null),
    [canSave, recording?.url, recording?.durationSeconds, fusion, features?.features, affect, insights],
  );

  if (!profileId) return null;

  return (
    <section className="longitudinal-panel panel-card">
      <div className="result-heading">
        <div>
          <p className="eyebrow">PHASE 21 · LONGITUDINAL VOICE INTELLIGENCE</p>
          <h3>See how this voice changes across sessions</h3>
          <p className="muted-copy">
            Save compact analysis summaries to build a profile-linked history of expressed vocal behavior. CEREBRO stores measurements and predictions here, not the raw recording.
          </p>
        </div>
        <span className={`result-badge ${history?.session_count ? "success" : "warning"}`}>
          {history?.session_count ? `${history.session_count} SESSIONS` : "NO HISTORY"}
        </span>
      </div>

      <div className="longitudinal-action-row">
        <button className="primary-button" type="button" onClick={saveCurrentSession} disabled={!canSave || busy || Boolean(savedSessionKey)}>
          {busy ? "Working…" : savedSessionKey ? "Saved to history" : "Save this analysis"}
        </button>
        <button className="text-button" type="button" onClick={refresh} disabled={busy || !history}>
          Refresh history
        </button>
        {history?.session_count > 0 && (
          <button className="text-button danger-text" type="button" onClick={clearHistory} disabled={busy}>
            Delete history
          </button>
        )}
      </div>

      {message && <p className="longitudinal-message">{message}</p>}
      {error && <p className="baseline-error" role="alert">{error}</p>}

      {!history?.session_count ? (
        <div className="longitudinal-empty">
          <strong>Build a longitudinal record</strong>
          <span>Analyze different natural recordings over time, then save the summaries you want CEREBRO to remember.</span>
        </div>
      ) : (
        <>
          <div className="longitudinal-metrics">
            <div className="metric-item"><span>Latest emotion</span><strong>{history.trends.latest_emotion || "—"}</strong></div>
            <div className="metric-item"><span>Latest confidence</span><strong>{history.trends.latest_confidence == null ? "—" : `${(history.trends.latest_confidence * 100).toFixed(1)}%`}</strong></div>
            <div className="metric-item"><span>Time span</span><strong>{history.trends.time_span_days.toFixed(1)} d</strong></div>
            <div className="metric-item"><span>Affect stability</span><strong>{history.trends.affect_stability == null ? "—" : `${(history.trends.affect_stability * 100).toFixed(0)}%`}</strong></div>
            <div className="metric-item"><span>Acoustic consistency</span><strong>{history.trends.vocal_consistency == null ? "—" : `${(history.trends.vocal_consistency * 100).toFixed(0)}%`}</strong></div>
            <div className="metric-item"><span>Avg quality</span><strong>{history.trends.average_quality_score == null ? "—" : `${(history.trends.average_quality_score * 100).toFixed(0)}%`}</strong></div>
          </div>

          <div className="longitudinal-grid">
            <div className="longitudinal-card">
              <div className="feature-section-heading"><span>AFFECT TRENDS</span><small>early → recent</small></div>
              <div className="longitudinal-trend-list">
                {trendCards.map((item) => (
                  <div className="longitudinal-trend-row" key={item.metric}>
                    <div><strong>{item.metric}</strong><small>{item.direction.replaceAll("_", " ")}</small></div>
                    <span>{formatDelta(item.delta_early_to_recent)}</span>
                    <i style={{ width: `${Math.max(6, item.strength * 100)}%` }} />
                  </div>
                ))}
              </div>
            </div>

            <div className="longitudinal-card">
              <div className="feature-section-heading"><span>EMOTION DISTRIBUTION</span><small>saved sessions</small></div>
              <div className="longitudinal-emotion-list">
                {EMOTIONS.map((emotion) => {
                  const item = history.trends.emotion_distribution.find((row) => row.emotion === emotion);
                  const share = item?.share || 0;
                  return (
                    <div className="emotion-row" key={emotion}>
                      <span>{emotion}</span>
                      <div className="emotion-track"><i style={{ width: `${Math.max(2, share * 100)}%` }} /></div>
                      <strong>{(share * 100).toFixed(0)}%</strong>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="longitudinal-card">
            <div className="feature-section-heading"><span>SESSION HISTORY</span><small>latest {Math.min(history.sessions.length, 8)}</small></div>
            <div className="longitudinal-session-list">
              {history.sessions.slice(-8).reverse().map((session) => (
                <div className="longitudinal-session-row" key={session.session_id}>
                  <span>{formatDate(session.recorded_at)}</span>
                  <strong>{session.emotion || "not classified"}</strong>
                  <span>{session.duration_seconds.toFixed(1)}s</span>
                  <span>{session.confidence == null ? "—" : `${(session.confidence * 100).toFixed(0)}%`}</span>
                  <span>V {session.affect.valence == null ? "—" : session.affect.valence.toFixed(2)}</span>
                  <span>A {session.affect.arousal == null ? "—" : session.affect.arousal.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </div>

          <p className="longitudinal-summary">{history.trends.summary}</p>
          <p className="model-note">{history.trends.caveat}</p>
        </>
      )}

      <VoiceChangePanel profileId={profileId} currentSession={changeSessionPayload} />
      <CrossSessionPatternsPanel profileId={profileId} refreshKey={patternRefreshKey} />
    </section>
  );
}
