import { useEffect, useRef, useState } from "react";
import { LiveAudioStreamer } from "../audio/liveStreamer";

function percent(value) {
  return value == null ? "—" : `${(Number(value) * 100).toFixed(1)}%`;
}

function formatSeconds(value) {
  return `${Number(value || 0).toFixed(1)}s`;
}

function titleCase(value) {
  return String(value || "—").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function speakerTone(index) {
  return `live-speaker-tone live-speaker-tone-${index % 6}`;
}

export default function LiveMultiSpeakerPanel() {
  const streamerRef = useRef(null);
  const [status, setStatus] = useState("idle");
  const [level, setLevel] = useState(0);
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState("");
  const startedAtRef = useRef(0);

  useEffect(() => () => streamerRef.current?.cancel(), []);

  useEffect(() => {
    if (status !== "live") return undefined;
    const timer = window.setInterval(() => {
      if (startedAtRef.current) setElapsed((performance.now() - startedAtRef.current) / 1000);
    }, 100);
    return () => window.clearInterval(timer);
  }, [status]);

  async function start() {
    setError("");
    setLatest(null);
    setHistory([]);
    setElapsed(0);
    setStatus("connecting");
    const streamer = new LiveAudioStreamer({
      maxDurationSeconds: 180,
      socketPath: "/audio/live-multi/ws",
      onLevel: setLevel,
      onMessage: (payload) => {
        if (payload.type === "ready") {
          startedAtRef.current = performance.now();
          setStatus("live");
        } else if (payload.type === "analysis") {
          setLatest(payload);
          setHistory((items) => [...items.slice(-9), payload]);
        } else if (payload.type === "error") {
          setError(payload.message || "Live multi-speaker analysis failed.");
        } else if (payload.type === "complete") {
          setStatus("complete");
          setElapsed(payload.duration_seconds || 0);
        }
      },
    });
    streamerRef.current = streamer;
    try {
      await streamer.start();
    } catch (err) {
      streamerRef.current = null;
      setStatus("idle");
      setError(err?.message || "Could not start live multi-speaker analysis.");
    }
  }

  async function stop() {
    if (!streamerRef.current) return;
    setStatus("stopping");
    setError("");
    try {
      await streamerRef.current.stop();
    } catch (err) {
      setStatus("idle");
      setError(err?.message || "Could not stop live multi-speaker analysis.");
    } finally {
      streamerRef.current = null;
    }
  }

  const speakers = latest?.speakers || [];
  const edges = latest?.interaction_graph?.edges || [];
  const dynamics = latest?.conversation_dynamics || {};
  const conversationState = latest?.conversation_state || {};
  const active = latest?.active_speakers || [];

  return (
    <section className="live-multi-card">
      <div className="section-header-row">
        <div>
          <p className="eyebrow">LIVE · REAL-TIME MULTI-SPEAKER INTELLIGENCE</p>
          <h2>Live conversation intelligence</h2>
          <p className="muted-copy">
            CEREBRO keeps a rolling audio context, re-diarizes recent speech, stabilizes speaker identities, and updates each speaker&apos;s emotion, affect, interaction graph, and conversation state as the conversation unfolds.
          </p>
        </div>
        <span className={`status-pill ${status === "live" ? "recording" : status === "complete" ? "complete" : ""}`}>
          <span className="status-dot" />
          {status === "idle" && "READY"}
          {status === "connecting" && "CONNECTING"}
          {status === "live" && "LIVE MULTI"}
          {status === "stopping" && "FINALIZING"}
          {status === "complete" && "COMPLETE"}
        </span>
      </div>

      <div className="live-multi-control-strip">
        <div className="live-wave" aria-hidden="true">
          {Array.from({ length: 40 }, (_, index) => {
            const shape = 0.35 + (Math.sin(index * 0.71) * 0.5 + 0.5) * 0.65;
            return <i key={index} style={{ height: `${10 + level * 55 * shape}px` }} />;
          })}
        </div>
        <div className="live-timer">{formatSeconds(elapsed)}</div>
        <div className="live-actions">
          {!['live', 'connecting', 'stopping'].includes(status) && <button className="primary-button" type="button" onClick={start}>Start multi-speaker analysis</button>}
          {status === "live" && <button className="primary-button stop-button" type="button" onClick={stop}>Stop analysis</button>}
          {status === "connecting" && <span className="processing-message"><span className="spinner" />Opening multi-speaker channel…</span>}
          {status === "stopping" && <span className="processing-message"><span className="spinner" />Finishing the current rolling context…</span>}
        </div>
      </div>

      {latest ? (
        <>
          <div className="live-multi-metrics">
            <div className="metric-card"><span>Speakers</span><strong>{latest.speaker_count}</strong></div>
            <div className="metric-card"><span>Active now</span><strong>{active.length}</strong></div>
            <div className="metric-card"><span>Conversation state</span><strong>{titleCase(latest.final_state)}</strong></div>
            <div className="metric-card"><span>ID stability</span><strong>{percent(latest.speaker_id_stability)}</strong></div>
            <div className="metric-card"><span>Latest speaker</span><strong>{latest.latest_speaker || "—"}</strong></div>
            <div className="metric-card"><span>Inference</span><strong>{Number(latest.latency_ms || 0).toFixed(0)} ms</strong></div>
          </div>

          <div className="live-multi-speaker-grid">
            {speakers.map((speaker, index) => (
              <article className={`live-multi-speaker-card ${speakerTone(index)}`} key={speaker.speaker}>
                <div className="live-multi-speaker-topline">
                  <span className="speaker-badge">{speaker.speaker}</span>
                  <span>{percent(speaker.speaking_share)}</span>
                </div>
                <h3>{titleCase(speaker.latest_emotion || speaker.dominant_emotion)}</h3>
                <div className="speaker-summary-row"><span>Turns</span><strong>{speaker.turn_count}</strong></div>
                <div className="speaker-summary-row"><span>Speaking</span><strong>{formatSeconds(speaker.speaking_seconds)}</strong></div>
                <div className="speaker-summary-row"><span>Confidence</span><strong>{percent(speaker.mean_confidence)}</strong></div>
                {speaker.mean_valence != null && <div className="speaker-affect-row"><span>V</span><strong>{speaker.mean_valence.toFixed(2)}</strong><span>A</span><strong>{speaker.mean_arousal?.toFixed(2)}</strong><span>D</span><strong>{speaker.mean_dominance?.toFixed(2)}</strong></div>}
              </article>
            ))}
          </div>

          <div className="live-multi-layout">
            <div className="live-multi-panel">
              <p className="eyebrow">LIVE SPEAKER TIMELINE</p>
              <div className="live-multi-timeline">
                {(latest.turns || []).slice(-12).map((turn, index) => (
                  <div className={`live-multi-turn ${speakerTone(speakers.findIndex((speaker) => speaker.speaker === turn.speaker))}`} key={turn.turn_id || `${turn.speaker}-${turn.start_seconds}-${index}`}>
                    <span>{formatSeconds(turn.start_seconds)}</span>
                    <strong>{turn.speaker}</strong>
                    <em>{turn.status === "provisional" ? "Provisional" : titleCase(turn.emotion)}</em>
                  </div>
                ))}
              </div>
              <small className="live-multi-note">Speaker IDs are stabilized across rolling contexts. A recent boundary may be revised when more speech arrives.</small>
            </div>

            <div className="live-multi-panel">
              <p className="eyebrow">LIVE INTERACTION GRAPH</p>
              {edges.length ? (
                <div className="live-multi-edge-list">
                  {edges.slice(0, 8).map((edge) => (
                    <div className="live-multi-edge-row" key={`${edge.from_speaker}-${edge.to_speaker}`}>
                      <strong>{edge.from_speaker} → {edge.to_speaker}</strong>
                      <span>{edge.interaction_count} responses</span>
                      <small>{titleCase(edge.activated_shift_count > 0 ? "activated_shift" : edge.emotion_transition_count > 0 ? "emotion_transition" : "continuity")}</small>
                    </div>
                  ))}
                </div>
              ) : <p className="muted-copy">Waiting for cross-speaker responses.</p>}
              <div className="live-multi-dynamics">
                <span>Tension</span><strong>{percent(dynamics.escalation_score)}</strong>
                <span>Synchrony</span><strong>{dynamics.affect_synchrony == null ? "—" : percent(dynamics.affect_synchrony)}</strong>
                <span>State</span><strong>{titleCase(conversationState.points?.at(-1)?.state || latest.final_state)}</strong>
              </div>
            </div>
          </div>
        </>
      ) : (
        <div className="live-empty-panel">
          <strong>Waiting for live multi-speaker analysis</strong>
          <span>The first update is emitted after CEREBRO has a complete rolling context.</span>
        </div>
      )}

      {error && <div className="message error-message" role="alert">{error}</div>}

      <div className="phase-note">
        <span>LIVE MULTI-SPEAKER CONTRACT</span>
        <strong>Mic → PCM16 → WebSocket → rolling 8 s context → diarization → stable speaker IDs → per-speaker emotion/affect → live graph + conversation state</strong>
      </div>
    </section>
  );
}
