import { useEffect, useRef, useState } from "react";

import { BrowserAudioRecorder } from "../audio/recorder";
import { compareEmotionRecordings } from "../api";

const EMOTION_EMOJI = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😊",
  neutral: "😐",
  sad: "😢",
  surprised: "😮",
};

const FEATURE_LABELS = {
  pitch_mean_hz: "Mean pitch",
  pitch_std_hz: "Pitch variation",
  energy_mean: "Vocal energy",
  energy_std: "Energy variation",
  spectral_centroid_hz: "Spectral centroid",
  spectral_bandwidth_hz: "Spectral bandwidth",
  spectral_rolloff_hz: "Spectral rolloff",
  zero_crossing_rate: "Zero-crossing rate",
  estimated_syllable_rate_sps: "Speech rate",
  estimated_pause_ratio: "Pause ratio",
  speech_coverage: "Speech coverage",
};

function formatTime(seconds) {
  const total = Math.max(0, Math.floor(seconds));
  return `${String(Math.floor(total / 60)).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function formatFeatureValue(item, value) {
  if (item.unit === "Hz") return `${value.toFixed(1)} Hz`;
  if (item.unit === "per_second") return `${value.toFixed(2)} /s`;
  return value.toFixed(3);
}

function formatRelative(value) {
  const percent = value * 100;
  if (Math.abs(percent) < 0.05) return "0.0%";
  return `${percent > 0 ? "+" : ""}${percent.toFixed(1)}%`;
}

function initialSlot() {
  return {
    status: "idle",
    elapsed: 0,
    level: 0,
    recording: null,
    error: "",
  };
}

function CaptureSlot({ label, subtitle, value, onChange }) {
  const recorderRef = useRef(null);
  const timerRef = useRef(null);
  const audioUrlRef = useRef(null);

  useEffect(() => {
    return () => {
      window.clearInterval(timerRef.current);
      recorderRef.current?.cancel();
      if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    };
  }, []);

  function clearSlot() {
    window.clearInterval(timerRef.current);
    timerRef.current = null;
    recorderRef.current?.cancel();
    recorderRef.current = null;
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
    onChange(initialSlot());
  }

  async function start() {
    clearSlot();
    onChange({ ...initialSlot(), status: "requesting" });

    const recorder = new BrowserAudioRecorder({
      maxDurationSeconds: 60,
      onLevel: (level) => onChange((current) => ({ ...current, level })),
    });
    recorderRef.current = recorder;

    try {
      await recorder.start();
      onChange((current) => ({ ...current, status: "recording", elapsed: 0 }));
      timerRef.current = window.setInterval(() => {
        onChange((current) => ({ ...current, elapsed: Math.min(60, current.elapsed + 1) }));
      }, 1000);
    } catch (error) {
      recorder.cancel();
      recorderRef.current = null;
      onChange({
        ...initialSlot(),
        error: error?.name === "NotAllowedError"
          ? "Microphone permission was denied."
          : error?.message || "Microphone capture could not be started.",
      });
    }
  }

  async function stop() {
    onChange((current) => ({ ...current, status: "processing" }));
    window.clearInterval(timerRef.current);
    timerRef.current = null;

    try {
      const output = await recorderRef.current.stop();
      const url = URL.createObjectURL(output.blob);
      audioUrlRef.current = url;
      onChange({
        status: "ready",
        elapsed: output.durationSeconds,
        level: 0,
        recording: {
          blob: output.blob,
          url,
          durationSeconds: output.durationSeconds,
          sampleRate: output.sampleRate,
        },
        error: "",
      });
    } catch (error) {
      onChange({ ...initialSlot(), error: error?.message || "Recording could not be completed." });
    } finally {
      recorderRef.current = null;
    }
  }

  const busy = ["requesting", "processing"].includes(value.status);
  const recording = value.status === "recording";

  return (
    <div className="comparison-slot">
      <div className="comparison-slot-head">
        <div>
          <span className="comparison-slot-label">{label}</span>
          <strong>{subtitle}</strong>
        </div>
        <span className={`status-pill ${recording ? "recording" : value.recording ? "complete" : ""}`}>
          <span className="status-dot" />
          {recording ? "RECORDING" : value.recording ? "CAPTURED" : "READY"}
        </span>
      </div>

      <div className="comparison-wave" aria-label={`${label} microphone level`}>
        {Array.from({ length: 20 }, (_, index) => {
          const phase = Math.sin(index * 0.8) * 0.5 + 0.5;
          const height = recording ? 12 + value.level * (20 + phase * 54) : 12 + phase * 8;
          return <span key={index} style={{ height: `${height}px` }} />;
        })}
        <span className="comparison-timer">{formatTime(value.elapsed)}</span>
      </div>

      <div className="comparison-slot-actions">
        {!recording && !busy && !value.recording && (
          <button className="secondary-button" type="button" onClick={start}>
            <span className="record-icon" />
            Record {label}
          </button>
        )}
        {recording && (
          <button className="primary-button" type="button" onClick={stop}>
            <span className="stop-icon" />
            Stop {label}
          </button>
        )}
        {busy && <span className="processing-message"><span className="spinner" />Preparing {label}…</span>}
        {value.recording && !busy && !recording && (
          <>
            <button className="secondary-button" type="button" onClick={start}>Re-record {label}</button>
            <a className="text-button comparison-save" href={value.recording.url} download={`cerebro-${label.toLowerCase().replace(" ", "-")}.wav`}>Save WAV</a>
          </>
        )}
      </div>

      {value.recording && (
        <audio className="comparison-audio" controls src={value.recording.url} />
      )}
      {value.error && <div className="comparison-slot-error">{value.error}</div>}
    </div>
  );
}

function DistributionComparison({ items }) {
  const ordered = [...items].sort((a, b) => b.absolute_delta - a.absolute_delta);
  return (
    <div className="comparison-distribution">
      {ordered.map((item) => (
        <div className="comparison-prob-row" key={item.emotion}>
          <span>{EMOTION_EMOJI[item.emotion] || "•"} {item.emotion}</span>
          <div className="comparison-prob-bars">
            <i style={{ width: `${Math.max(2, item.a_probability * 100)}%` }} />
            <b style={{ width: `${Math.max(2, item.b_probability * 100)}%` }} />
          </div>
          <strong>{formatRelative(item.delta)}</strong>
        </div>
      ))}
      <div className="comparison-legend"><span><i /> A</span><span><b /> B</span></div>
    </div>
  );
}

function RecordingSummary({ title, result }) {
  const fusion = result.fusion;
  return (
    <div className="comparison-recording-summary">
      <span className="eyebrow">{title}</span>
      <div className="comparison-emotion-main">
        <span>{EMOTION_EMOJI[fusion.emotion] || "•"}</span>
        <div>
          <strong>{fusion.emotion}</strong>
          <small>{pct(fusion.confidence)} calibrated confidence</small>
        </div>
      </div>
      <div className="comparison-recording-metrics">
        <div><span>Duration</span><strong>{result.audio.duration_seconds.toFixed(1)} s</strong></div>
        <div><span>Energy</span><strong>{result.features.energy.mean.toFixed(4)}</strong></div>
        <div><span>Pitch</span><strong>{result.features.pitch_hz.mean.toFixed(1)} Hz</strong></div>
        <div><span>Speech rate</span><strong>{result.features.estimated_syllable_rate_sps.toFixed(2)} /s</strong></div>
      </div>
      <div className="comparison-model-line">
        <span>Classical <strong>{fusion.classical.emotion}</strong></span>
        <span>Deep <strong>{fusion.deep.emotion}</strong></span>
        <span>Agreement <strong>{fusion.agreement}</strong></span>
      </div>
    </div>
  );
}

export default function RecordingComparisonLab() {
  const [slotA, setSlotA] = useState(initialSlot());
  const [slotB, setSlotB] = useState(initialSlot());
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [labKey, setLabKey] = useState(0);

  function resetLab() {
    setSlotA(initialSlot());
    setSlotB(initialSlot());
    setResult(null);
    setStatus("idle");
    setError("");
    setLabKey((current) => current + 1);
  }

  async function compare() {
    if (!slotA.recording || !slotB.recording) return;
    setStatus("comparing");
    setError("");
    try {
      const payload = await compareEmotionRecordings(slotA.recording.blob, slotB.recording.blob);
      setResult(payload);
      setStatus("complete");
    } catch (compareError) {
      setStatus("idle");
      setError(compareError?.message || "Comparison failed.");
    }
  }

  const ready = Boolean(slotA.recording && slotB.recording);
  const comparison = result?.comparison;
  const topFeatures = result?.acoustic_features?.slice().sort((a, b) => Math.abs(b.relative_delta) - Math.abs(a.relative_delta)).slice(0, 6) || [];

  return (
    <section className="comparison-lab">
      <div className="comparison-lab-header">
        <div>
          <p className="eyebrow">PHASE 10 · RECORDING COMPARISON + CONTENT VERIFICATION</p>
          <h2>Say it twice. Change the tone.</h2>
          <p className="muted-copy">
            Speak the same sentence in both takes and deliberately change only the delivery. CEREBRO runs the same calibrated inference stack on each recording, then measures what changed acoustically and emotionally.
          </p>
        </div>
        <span className="build-chip">CONTROLLED COMPARISON</span>
      </div>

      <div className="comparison-protocol">
        <div className="protocol-step"><span>01</span><strong>Same words</strong><small>Keep the sentence identical.</small></div>
        <div className="protocol-arrow">→</div>
        <div className="protocol-step"><span>02</span><strong>Change delivery</strong><small>Try calm vs. angry, happy vs. sad, etc.</small></div>
        <div className="protocol-arrow">→</div>
        <div className="protocol-step"><span>03</span><strong>Compare evidence</strong><small>Inspect acoustic and emotion shifts.</small></div>
      </div>

      <div className="comparison-slot-grid">
        <CaptureSlot key={`a-${labKey}`} label="Recording A" subtitle="First delivery" value={slotA} onChange={setSlotA} />
        <CaptureSlot key={`b-${labKey}`} label="Recording B" subtitle="Changed delivery" value={slotB} onChange={setSlotB} />
      </div>

      <div className="comparison-actions">
        <button className="primary-button" type="button" disabled={!ready || status === "comparing"} onClick={compare}>
          {status === "comparing" ? <><span className="spinner" /> Comparing…</> : <>Compare recordings</>}
        </button>
        {result && <button className="secondary-button" type="button" onClick={resetLab}>Start another comparison</button>}
      </div>

      {error && <div className="message error-message" role="alert">{error}</div>}

      {result && (
        <div className="comparison-result">
          <div className="result-heading">
            <div>
              <p className="eyebrow">COMPARISON RESULT</p>
              <h3>{comparison.emotion_changed ? "The conveyed-emotion estimate changed." : "The final emotion estimate stayed the same."}</h3>
              <p className="muted-copy">{comparison.interpretation}</p>
            </div>
            <span className={`result-badge ${comparison.emotion_changed ? "success" : "warning"}`}>
              {comparison.emotion_changed ? "EMOTION SHIFT" : "SAME TOP EMOTION"}
            </span>
          </div>

          <div className="comparison-score-grid">
            <div className="comparison-score-card"><span>Acoustic change</span><strong>{pct(comparison.acoustic_change_score)}</strong><small>measured feature contrast</small></div>
            <div className="comparison-score-card"><span>Acoustic similarity</span><strong>{pct(comparison.acoustic_similarity_score)}</strong><small>overall profile similarity</small></div>
            <div className="comparison-score-card"><span>Emotion shift</span><strong>{comparison.emotion_distribution_shift.toFixed(3)}</strong><small>Jensen–Shannon divergence</small></div>
            <div className="comparison-score-card"><span>Largest probability movement</span><strong>{comparison.top_emotion_shift}</strong><small>highest absolute change</small></div>
          </div>

          <div className="comparison-summary-grid">
            <RecordingSummary title="RECORDING A" result={result.recording_a} />
            <RecordingSummary title="RECORDING B" result={result.recording_b} />
          </div>

          <div className="comparison-panel">
            <div className="feature-section-heading">
              <span>EMOTION DISTRIBUTION SHIFT</span>
              <small>positive means B gained probability</small>
            </div>
            <DistributionComparison items={result.emotion_probabilities} />
          </div>

          <div className="comparison-panel">
            <div className="feature-section-heading">
              <span>WHAT CHANGED ACOUSTICALLY</span>
              <small>largest relative movements first</small>
            </div>
            <div className="comparison-feature-table">
              {topFeatures.map((item) => (
                <div className="comparison-feature-row" key={item.name}>
                  <div>
                    <strong>{FEATURE_LABELS[item.name] || item.name.replaceAll("_", " ")}</strong>
                    <small>{item.direction} · {item.unit}</small>
                  </div>
                  <span>{formatFeatureValue(item, item.a_value)}</span>
                  <span>→</span>
                  <span>{formatFeatureValue(item, item.b_value)}</span>
                  <strong className={item.absolute_delta >= 0 ? "delta-up" : "delta-down"}>{formatRelative(item.relative_delta)}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="comparison-panel content-verification-panel">
            <div className="feature-section-heading">
              <span>CONTENT VERIFICATION</span>
              <small>local Whisper ASR · normalized word-sequence comparison</small>
            </div>
            {result.content_verification?.asr_available ? (
              <>
                <div className="content-verification-status-row">
                  <span className={`result-badge ${result.content_verification.status === "same" ? "success" : result.content_verification.status === "uncertain" ? "warning" : "error"}`}>
                    {result.content_verification.status.toUpperCase()} CONTENT
                  </span>
                  <div className="content-verification-metrics">
                    <span>Similarity <strong>{pct(result.content_verification.similarity)}</strong></span>
                    <span>WER <strong>{(result.content_verification.word_error_rate * 100).toFixed(1)}%</strong></span>
                    <span>A <strong>{result.content_verification.words_a} words</strong></span>
                    <span>B <strong>{result.content_verification.words_b} words</strong></span>
                  </div>
                </div>
                <div className="transcript-grid">
                  <div className="transcript-card"><span>TRANSCRIPT A</span><p>{result.content_verification.transcript_a || "[no transcription]"}</p></div>
                  <div className="transcript-card"><span>TRANSCRIPT B</span><p>{result.content_verification.transcript_b || "[no transcription]"}</p></div>
                </div>
                <p className="muted-copy content-verification-interpretation">{result.content_verification.interpretation}</p>
              </>
            ) : (
              <div className="message warning-message">
                <strong>Content verification unavailable.</strong> {result.content_verification?.interpretation || "Local transcription is not available."}
              </div>
            )}
          </div>

          <div className="comparison-note">
            <strong>Experimental protocol:</strong> {result.protocol_note}
          </div>
        </div>
      )}
    </section>
  );
}
