import { useEffect, useRef, useState } from "react";
import { LiveAudioStreamer } from "../audio/liveStreamer";

function percent(value) {
  return value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
}

function signed(value) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(2)}`;
}

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds || 0)));
  const minutes = Math.floor(total / 60).toString().padStart(2, "0");
  const secs = (total % 60).toString().padStart(2, "0");
  return `${minutes}:${secs}`;
}

function titleCase(value) {
  return String(value || "—")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export default function LiveEmotionPanel() {
  const streamerRef = useRef(null);
  const [status, setStatus] = useState("idle");
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState("");
  const [complete, setComplete] = useState(null);

  useEffect(() => {
    let interval;
    if (status === "live") {
      const started = performance.now();
      interval = window.setInterval(() => {
        setElapsed((performance.now() - started) / 1000);
      }, 100);
    }
    return () => window.clearInterval(interval);
  }, [status]);

  useEffect(() => () => streamerRef.current?.cancel(), []);

  async function startLive() {
    setError("");
    setLatest(null);
    setHistory([]);
    setComplete(null);
    setElapsed(0);
    setStatus("connecting");

    const streamer = new LiveAudioStreamer({
      maxDurationSeconds: 120,
      onLevel: setLevel,
      onMessage: (payload) => {
        if (payload.type === "ready") {
          setStatus("live");
        } else if (payload.type === "analysis") {
          setLatest(payload);
          setHistory((current) => [...current.slice(-9), payload]);
        } else if (payload.type === "error") {
          setError(payload.message || "Live analysis failed.");
        } else if (payload.type === "complete") {
          setComplete(payload);
          setStatus("complete");
          setElapsed(payload.duration_seconds || 0);
        }
      },
    });
    streamerRef.current = streamer;

    try {
      await streamer.start();
    } catch (err) {
      setStatus("idle");
      setError(err?.message || "Could not start live analysis.");
      streamerRef.current = null;
    }
  }

  async function stopLive() {
    if (!streamerRef.current) return;
    setStatus("stopping");
    setError("");
    try {
      await streamerRef.current.stop();
    } catch (err) {
      setStatus("idle");
      setError(err?.message || "Could not stop live analysis.");
    } finally {
      streamerRef.current = null;
    }
  }

  const probabilityItems = latest
    ? Object.entries(latest.probabilities || {}).sort((a, b) => b[1] - a[1]).slice(0, 7)
    : [];

  return (
    <section className="live-emotion-card">
      <div className="section-header-row">
        <div>
          <p className="eyebrow">PHASE 20 · REAL-TIME SPEECH EMOTION</p>
          <h2>Live CEREBRO</h2>
          <p className="muted-copy">
            Speak naturally. CEREBRO streams short PCM16 audio chunks over one WebSocket session and analyzes rolling three-second windows. This mode is currently single-speaker; multi-speaker diarization remains in the recorded-conversation workflow.
          </p>
        </div>
        <span className={`status-pill ${status === "live" ? "recording" : status === "complete" ? "complete" : ""}`}>
          <span className="status-dot" />
          {status === "idle" && "READY"}
          {status === "connecting" && "CONNECTING"}
          {status === "live" && "LIVE"}
          {status === "stopping" && "FINALIZING"}
          {status === "complete" && "COMPLETE"}
        </span>
      </div>

      <div className="live-control-strip">
        <div className="live-wave" aria-hidden="true">
          {Array.from({ length: 36 }, (_, index) => {
            const shape = 0.35 + (Math.sin(index * 0.71) * 0.5 + 0.5) * 0.65;
            const height = 10 + level * 55 * shape;
            return <i key={index} style={{ height: `${height}px` }} />;
          })}
        </div>
        <div className="live-timer">{formatTime(elapsed)}</div>
        <div className="live-actions">
          {!["live", "connecting", "stopping"].includes(status) && (
            <button className="primary-button" type="button" onClick={startLive}>Start live analysis</button>
          )}
          {status === "live" && (
            <button className="primary-button stop-button" type="button" onClick={stopLive}>Stop live analysis</button>
          )}
          {status === "connecting" && <span className="processing-message"><span className="spinner" />Opening live analysis channel…</span>}
          {status === "stopping" && <span className="processing-message"><span className="spinner" />Finishing the current analysis window…</span>}
        </div>
      </div>

      {latest ? (
        <div className="live-result-grid">
          <div className="live-primary-panel">
            <p className="eyebrow">LATEST ROLLING WINDOW</p>
            <div className="live-emotion-value">{latest.emotion ? latest.emotion.toUpperCase() : "QUIET"}</div>
            <div className="live-confidence">{percent(latest.confidence)} confidence</div>
            <div className="live-window-meta">
              {Number(latest.window_start_seconds).toFixed(1)}s → {Number(latest.window_end_seconds).toFixed(1)}s
              <span>·</span>
              {Number(latest.latency_ms).toFixed(0)} ms inference
            </div>
          </div>

          <div className="live-state-panel">
            <div className="live-state-row"><span>Vocal state</span><strong>{titleCase(latest.state)}</strong></div>
            <div className="live-state-row"><span>Valence</span><strong>{signed(latest.valence)}</strong></div>
            <div className="live-state-row"><span>Arousal</span><strong>{signed(latest.arousal)}</strong></div>
            <div className="live-state-row"><span>Dominance</span><strong>{signed(latest.dominance)}</strong></div>
            <div className="live-state-row"><span>Audio quality</span><strong>{titleCase(latest.quality_verdict)}</strong></div>
          </div>

          <div className="live-probability-panel">
            <p className="eyebrow">LIVE PROBABILITY DISTRIBUTION</p>
            {probabilityItems.map(([label, value]) => (
              <div className="emotion-row" key={label}>
                <span>{label}</span>
                <div className="emotion-track"><i style={{ width: `${Math.max(2, value * 100)}%` }} /></div>
                <strong>{percent(value)}</strong>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="live-empty-panel">
          <strong>Waiting for live speech analysis</strong>
          <span>The first prediction is emitted after CEREBRO collects one complete rolling analysis window.</span>
        </div>
      )}

      {history.length > 0 && (
        <div className="live-history-panel">
          <div className="section-header-row compact">
            <div>
              <p className="eyebrow">REAL-TIME TRAJECTORY</p>
              <h3>Recent live states</h3>
            </div>
            <span className="muted-copy">{history.length} windows</span>
          </div>
          <div className="live-history-list">
            {history.map((item) => (
              <div className="live-history-row" key={item.sequence}>
                <span>{Number(item.window_end_seconds).toFixed(1)}s</span>
                <strong>{item.emotion ? titleCase(item.emotion) : "Quiet"}</strong>
                <span>{percent(item.confidence)}</span>
                <span>{titleCase(item.state)}</span>
                {item.state_transition && <em>{item.state_transition.replaceAll("_", " ")}</em>}
              </div>
            ))}
          </div>
        </div>
      )}

      {complete && (
        <div className="message complete-message" role="status">
          <span className="eyebrow">LIVE SESSION COMPLETE</span>
          <strong>{Number(complete.updates || 0)} rolling analysis updates</strong>
          <span>{Number(complete.duration_seconds || 0).toFixed(1)} seconds captured · final vocal state: {titleCase(complete.final_state)}</span>
        </div>
      )}

      {error && (
        <div className="message error-message" role="alert">
          {error}
        </div>
      )}

      <div className="phase-note">
        <span>REAL-TIME CONTRACT</span>
        <strong>Microphone → AudioWorklet → PCM16 chunks → WebSocket → rolling 3 s window → calibrated fusion + affect → live vocal state</strong>
      </div>
    </section>
  );
}
