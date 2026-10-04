import { useEffect, useRef, useState } from "react";

import { analyzeSpeakerAware } from "../api";
import ConversationInteractionGraph from "./ConversationInteractionGraph";
import ConversationDynamicsPanel from "./ConversationDynamicsPanel";
import ConversationStatePanel from "./ConversationStatePanel";

function formatSeconds(value) {
  return `${Number(value || 0).toFixed(1)}s`;
}

function titleCase(value) {
  return String(value || "").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function speakerClass(index) {
  return `speaker-track speaker-track-${index % 6}`;
}

export default function SpeakerAwareAnalysis() {
  const [blob, setBlob] = useState(null);
  const [filename, setFilename] = useState("cerebro-conversation.wav");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const inputRef = useRef(null);

  useEffect(() => {
    function onRecording(event) {
      if (event.detail instanceof Blob) {
        setBlob(event.detail);
        setFilename("cerebro-last-recording.wav");
        setResult(null);
        setError("");
      }
    }
    window.addEventListener("cerebro:recording-ready", onRecording);
    return () => window.removeEventListener("cerebro:recording-ready", onRecording);
  }, []);

  async function analyze() {
    if (!blob) {
      setError("Add a multi-speaker WAV recording first.");
      return;
    }
    setStatus("analyzing");
    setError("");
    setResult(null);
    try {
      const response = await analyzeSpeakerAware(blob, filename);
      setResult(response);
      setStatus("complete");
    } catch (err) {
      setError(err?.message || "Speaker-aware analysis failed.");
      setStatus("error");
    }
  }

  function handleFile(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    setBlob(file);
    setFilename(file.name || "cerebro-conversation.wav");
    setResult(null);
    setError("");
    setStatus("idle");
  }

  return (
    <section className="speaker-lab-card">
      <div className="section-header-row">
        <div>
          <p className="eyebrow">MULTI-SPEAKER INTELLIGENCE</p>
          <h2>Track emotion by speaker</h2>
          <p className="muted-copy">
            Upload a conversation with multiple voices. CEREBRO diarizes the recording,
            analyzes each speaker turn independently, and reconstructs speaker-level emotional trajectories.
          </p>
        </div>
        <div className={`status-pill ${status}`}>
          <span className="status-dot" />
          {status === "idle" && "READY"}
          {status === "analyzing" && "DIARIZING + ANALYZING"}
          {status === "complete" && "ANALYSIS COMPLETE"}
          {status === "error" && "NEEDS ATTENTION"}
        </div>
      </div>

      <div className="speaker-input-row">
        <label className="file-input-label">
          <span>Select conversation WAV</span>
          <input ref={inputRef} type="file" accept="audio/wav,audio/x-wav,audio/wave" onChange={handleFile} />
        </label>
        <button className="primary-button" type="button" onClick={analyze} disabled={!blob || status === "analyzing"}>
          {status === "analyzing" ? "Analyzing…" : "Analyze speakers"}
        </button>
      </div>

      {blob && !result && (
        <div className="speaker-ready-state">
          <strong>{filename}</strong>
          <span>Ready for diarization and speaker-level emotion analysis.</span>
        </div>
      )}

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <div className="speaker-results">
          <div className="metrics-grid">
            <div className="metric-card"><span>Speakers</span><strong>{result.speaker_count}</strong></div>
            <div className="metric-card"><span>Speaker switches</span><strong>{result.speaker_switches}</strong></div>
            <div className="metric-card"><span>Speech coverage</span><strong>{Math.round(result.speech_coverage * 100)}%</strong></div>
            <div className="metric-card"><span>Overlap</span><strong>{formatSeconds(result.overlap_seconds)}</strong></div>
          </div>

          <div className="speaker-summary-grid">
            {result.speakers.map((speaker, index) => (
              <article className={`speaker-summary-card ${speakerClass(index)}`} key={speaker.speaker}>
                <div className="speaker-summary-topline">
                  <span className="speaker-badge">{speaker.speaker}</span>
                  <span>{Math.round(speaker.speaking_share * 100)}% of active speech</span>
                </div>
                <h3>{titleCase(speaker.dominant_emotion)}</h3>
                <div className="speaker-summary-row"><span>Turns</span><strong>{speaker.turn_count}</strong></div>
                <div className="speaker-summary-row"><span>Speaking time</span><strong>{formatSeconds(speaker.speaking_seconds)}</strong></div>
                <div className="speaker-summary-row"><span>Mean confidence</span><strong>{Math.round(speaker.mean_confidence * 100)}%</strong></div>
                <div className="speaker-summary-row"><span>First → last</span><strong>{titleCase(speaker.first_emotion)} → {titleCase(speaker.last_emotion)}</strong></div>
                {speaker.mean_valence != null && (
                  <div className="speaker-affect-row">
                    <span>V</span><strong>{speaker.mean_valence.toFixed(2)}</strong>
                    <span>A</span><strong>{speaker.mean_arousal.toFixed(2)}</strong>
                    <span>D</span><strong>{speaker.mean_dominance.toFixed(2)}</strong>
                  </div>
                )}
              </article>
            ))}
          </div>

          <div className="speaker-timeline-card">
            <div className="section-header-row compact">
              <div>
                <p className="eyebrow">SPEAKER TIMELINE</p>
                <h3>Who spoke, when, and with what conveyed emotion</h3>
              </div>
              <span className="muted-copy">{result.diarization_model}</span>
            </div>
            <div className="speaker-timeline">
              {result.turns.map((turn, index) => (
                <div
                  className={`speaker-turn ${speakerClass(result.speakers.findIndex((item) => item.speaker === turn.speaker))}`}
                  key={`${turn.turn_index}-${turn.speaker}-${turn.start_seconds}`}
                  style={{ left: `${(turn.start_seconds / result.duration_seconds) * 100}%`, width: `${Math.max(0.7, (turn.duration_seconds / result.duration_seconds) * 100)}%` }}
                  title={`${turn.speaker}: ${titleCase(turn.emotion)} ${Math.round(turn.confidence * 100)}%`}
                >
                  <span>{turn.speaker}</span>
                  <strong>{titleCase(turn.emotion)}</strong>
                </div>
              ))}
            </div>
          </div>

          <ConversationInteractionGraph graph={result.interaction_graph} />
          <ConversationDynamicsPanel dynamics={result.conversation_dynamics} />
          <ConversationStatePanel state={result.conversation_state} />

          <div className="speaker-interactions-card">
            <p className="eyebrow">CROSS-SPEAKER INTERACTIONS</p>
            <h3>Turn-to-turn emotional changes</h3>
            {result.interactions.length === 0 ? (
              <p className="muted-copy">Only one speaker produced a usable sequence, so no cross-speaker transitions were available.</p>
            ) : (
              <div className="interaction-list">
                {result.interactions.map((item, index) => (
                  <div className="interaction-item" key={`${item.at_seconds}-${index}`}>
                    <span>{formatSeconds(item.at_seconds)}</span>
                    <strong>{item.from_speaker} → {item.to_speaker}</strong>
                    <span>{titleCase(item.previous_emotion)} → {titleCase(item.next_emotion)}</span>
                    <em>{item.type.replaceAll("_", " ")}</em>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
