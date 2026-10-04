import { useEffect, useRef, useState } from "react";

import { BrowserAudioRecorder } from "../audio/recorder";
import TemporalEmotionTimeline from "./TemporalEmotionTimeline";
import EmotionInsights from "./EmotionInsights";
import AffectSpacePanel from "./AffectSpacePanel";
import AffectDynamicsPanel from "./AffectDynamicsPanel";
import PersonalBaselinePanel from "./PersonalBaselinePanel";
import LongitudinalVoicePanel from "./LongitudinalVoicePanel";
import PrivacyPanel from "./PrivacyPanel";
import { predictAffectDynamics, predictDimensionalAffect, predictEmotionInsights, predictTemporalEmotion, transcribeAudio, validateAudio } from "../api";

function formatTime(seconds) {
  const value = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(value / 60).toString().padStart(2, "0");
  const remaining = (value % 60).toString().padStart(2, "0");
  return `${minutes}:${remaining}`;
}

export default function AudioRecorder() {
  const recorderRef = useRef(null);
  const timerRef = useRef(null);
  const audioUrlRef = useRef(null);

  const [status, setStatus] = useState("idle");
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);
  const [recording, setRecording] = useState(null);
  const [result, setResult] = useState(null);
  const [featureResult, setFeatureResult] = useState(null);
  const [insightsResult, setInsightsResult] = useState(null);
  const [fusionResult, setFusionResult] = useState(null);
  const [fusionError, setFusionError] = useState("");
  const [temporalResult, setTemporalResult] = useState(null);
  const [temporalError, setTemporalError] = useState("");
  const [transcriptionResult, setTranscriptionResult] = useState(null);
  const [affectResult, setAffectResult] = useState(null);
  const [affectError, setAffectError] = useState("");
  const [affectDynamicsResult, setAffectDynamicsResult] = useState(null);
  const [affectDynamicsError, setAffectDynamicsError] = useState("");
  const [transcriptionError, setTranscriptionError] = useState("");
  const [error, setError] = useState("");
  const [profileId, setProfileId] = useState(null);

  useEffect(() => {
    return () => {
      window.clearInterval(timerRef.current);
      recorderRef.current?.cancel();
      if (audioUrlRef.current) {
        URL.revokeObjectURL(audioUrlRef.current);
      }
    };
  }, []);

  function clearResult() {
    setResult(null);
    setFeatureResult(null);
    setInsightsResult(null);
    setFusionResult(null);
    setFusionError("");
    setTemporalResult(null);
    setTemporalError("");
    setTranscriptionResult(null);
    setTranscriptionError("");
    setAffectResult(null);
    setAffectError("");
    setAffectDynamicsResult(null);
    setAffectDynamicsError("");
    setError("");
    setRecording(null);
    if (audioUrlRef.current) {
      URL.revokeObjectURL(audioUrlRef.current);
      audioUrlRef.current = null;
    }
  }

  async function startRecording() {
    clearResult();
    setStatus("requesting");

    const recorder = new BrowserAudioRecorder({
      maxDurationSeconds: 60,
      onLevel: setLevel,
    });
    recorderRef.current = recorder;

    try {
      await recorder.start();
      setStatus("recording");
      setElapsed(0);
      timerRef.current = window.setInterval(() => {
        setElapsed((current) => {
          if (current >= 59) {
            return 60;
          }
          return current + 1;
        });
      }, 1000);
    } catch (err) {
      recorder.cancel();
      setStatus("idle");
      setError(
        err?.name === "NotAllowedError"
          ? "Microphone permission was denied. Allow microphone access and try again."
          : err?.message || "Microphone capture could not be started."
      );
    }
  }

  async function stopRecording() {
    setStatus("processing");
    window.clearInterval(timerRef.current);
    timerRef.current = null;

    try {
      const output = await recorderRef.current.stop();
      const url = URL.createObjectURL(output.blob);
      audioUrlRef.current = url;
      setRecording({
        sessionId: typeof window.crypto?.randomUUID === "function"
          ? window.crypto.randomUUID()
          : `session-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`,
        blob: output.blob,
        url,
        durationSeconds: output.durationSeconds,
        sampleRate: output.sampleRate,
      });
      window.dispatchEvent(new CustomEvent("cerebro:recording-ready", { detail: output.blob }));

      setStatus("validating");
      const response = await validateAudio(output.blob);
      setResult(response);

      setStatus("inferring");
      try {
        const insights = await predictEmotionInsights(output.blob, undefined, profileId);
        setInsightsResult(insights);
        setFeatureResult({
          audio: insights.audio,
          features: insights.features,
          next_step: "Feature extraction is now feeding calibrated fusion and the Phase 7 explanation layer.",
        });
        setFusionResult(insights.fusion);
        setFusionError("");
      } catch (insightErr) {
        setFusionError(
          insightErr?.message ||
          "Calibrated fusion and explanation are not available yet. Train the Phase 5 calibration artifact."
        );
      }

      setStatus("affect");
      try {
        const affect = await predictDimensionalAffect(output.blob);
        setAffectResult(affect);
        setAffectError("");
      } catch (affectErr) {
        setAffectError(affectErr?.message || "Continuous affect analysis is not available yet.");
      }

      setStatus("dynamics");
      try {
        const dynamics = await predictAffectDynamics(output.blob);
        setAffectDynamicsResult(dynamics);
        setAffectDynamicsError("");
      } catch (dynamicsErr) {
        setAffectDynamicsError(
          dynamicsErr?.message ||
          "Affect dynamics analysis is not available yet."
        );
      }

      setStatus("transcribing");
      try {
        const transcription = await transcribeAudio(output.blob);
        setTranscriptionResult(transcription);
        setTranscriptionError("");
      } catch (transcriptionErr) {
        setTranscriptionError(
          transcriptionErr?.message ||
          "Local speech transcription is not available yet."
        );
      }

      setStatus("tracking");
      try {
        const temporal = await predictTemporalEmotion(output.blob);
        setTemporalResult(temporal);
        setTemporalError("");
      } catch (temporalErr) {
        setTemporalError(
          temporalErr?.message ||
          "Temporal emotion analysis is not available yet."
        );
      }

      setStatus("complete");
    } catch (err) {
      setStatus("idle");
      setError(err?.message || "Audio processing failed.");
    } finally {
      recorderRef.current = null;
    }
  }

  const isBusy = ["requesting", "processing", "validating", "inferring", "affect", "dynamics", "transcribing", "tracking"].includes(status);
  const isRecording = status === "recording";

  return (
    <section className="recording-card">
      <div className="recording-card-header">
        <div>
          <p className="eyebrow">SPEECH EMOTION ANALYSIS</p>
          <h2>Capture and analyze a real voice sample</h2>
          <p className="muted-copy">
            Speak naturally. CEREBRO captures PCM WAV in the browser, validates it,
            normalizes it to mono 16 kHz, and extracts a structured acoustic signature.
          </p>
        </div>
        <div className={`status-pill ${status}`}>
          <span className="status-dot" />
          {status === "idle" && "READY"}
          {status === "requesting" && "MIC PERMISSION"}
          {status === "recording" && "RECORDING"}
          {status === "processing" && "ENCODING"}
          {status === "validating" && "VALIDATING"}
                    {status === "inferring" && "RUNNING FUSION + EXPLAINABILITY"}
          {status === "affect" && "ESTIMATING VALENCE · AROUSAL · DOMINANCE"}
          {status === "dynamics" && "TRACKING VALENCE · AROUSAL · DOMINANCE THROUGH TIME"}
          {status === "transcribing" && "TRANSCRIBING SPEECH LOCALLY"}
          {status === "tracking" && "BUILDING THE EMOTION TRAJECTORY"}
          {status === "complete" && "ANALYSIS COMPLETE"}
        </div>
      </div>

      <div className="wave-stage" aria-label="Microphone level visualization">
        <div className="wave-glow" style={{ transform: `scaleX(${0.45 + level * 0.55})` }} />
        <div className="wave-bars" aria-hidden="true">
          {Array.from({ length: 32 }, (_, index) => {
            const phase = Math.sin(index * 0.82) * 0.5 + 0.5;
            const height = isRecording
              ? 18 + level * (22 + phase * 62)
              : 18 + phase * 10;
            return <span key={index} style={{ height: `${height}px` }} />;
          })}
        </div>
        <div className="timer">{formatTime(elapsed)}</div>
      </div>

      <div className="recording-controls">
        {!isRecording && !isBusy && (
          <button className="primary-button record-button" onClick={startRecording} type="button">
            <span className="record-icon" />
            Start recording
          </button>
        )}

        {isRecording && (
          <button className="primary-button stop-button" onClick={stopRecording} type="button">
            <span className="stop-icon" />
            Stop & analyze
          </button>
        )}

        {isBusy && (
          <div className="processing-message">
            <span className="spinner" />
            {status === "requesting" && "Requesting microphone access…"}
            {status === "processing" && "Turning the recording into a clean WAV…"}
            {status === "validating" && "Sending audio to the validation service…"}
                        {status === "inferring" && "Running the classical model, deep speech model, calibration, and fusion…"}
            {status === "affect" && "Estimating continuous valence, arousal, and dominance…"}
            {status === "dynamics" && "Tracking the emotional-state trajectory through overlapping speech windows…"}
            {status === "transcribing" && "Running local Whisper transcription for content-aware analysis…"}
            {status === "tracking" && "Analyzing overlapping windows and smoothing emotion probabilities…"}
          </div>
        )}

        {recording && status === "complete" && (
          <a className="secondary-button" href={recording.url} download="cerebro-recording.wav">
            Save WAV
          </a>
        )}
      </div>

      {recording && (
        <div className="playback-row">
          <span className="playback-label">Captured audio</span>
          <audio controls src={recording.url} />
        </div>
      )}

      <PersonalBaselinePanel recording={recording && status === "complete" ? recording : null} onProfileChange={setProfileId} />
      <LongitudinalVoicePanel
        profileId={profileId}
        recording={recording && status === "complete" ? recording : null}
        fusion={fusionResult}
        features={featureResult}
        affect={affectResult}
        insights={insightsResult}
      />
      <PrivacyPanel profileId={profileId} />

      {error && (
        <div className="message error-message" role="alert">
          {error}
        </div>
      )}

      {result && (
        <div className="validation-result">
          <div className="result-heading">
            <div>
              <p className="eyebrow">INGESTION CHECK</p>
              <h3>{result.audio.is_silent ? "Valid audio, but nearly silent" : "Audio accepted"}</h3>
            </div>
            <span className={`result-badge ${result.audio.is_silent ? "warning" : "success"}`}>
              {result.audio.is_silent ? "RECORD AGAIN" : "GREEN"}
            </span>
          </div>

          <div className="metric-grid">
            <div className="metric-item">
              <span>Duration</span>
              <strong>{result.audio.duration_seconds.toFixed(2)} s</strong>
            </div>
            <div className="metric-item">
              <span>Source rate</span>
              <strong>{(result.audio.original_sample_rate / 1000).toFixed(1)} kHz</strong>
            </div>
            <div className="metric-item">
              <span>Model rate</span>
              <strong>{(result.audio.sample_rate / 1000).toFixed(1)} kHz</strong>
            </div>
            <div className="metric-item">
              <span>Channels</span>
              <strong>{result.audio.original_channels} → {result.audio.channels}</strong>
            </div>
            <div className="metric-item">
              <span>RMS</span>
              <strong>{result.audio.rms.toFixed(4)}</strong>
            </div>
            <div className="metric-item">
              <span>Peak</span>
              <strong>{result.audio.peak.toFixed(4)}</strong>
            </div>
          </div>

          <p className="next-step">{result.next_step}</p>
        </div>
      )}

      {featureResult && (
        <div className="feature-result">
          <div className="result-heading">
            <div>
              <p className="eyebrow">ACOUSTIC SIGNATURE · PHASE 2</p>
              <h3>Acoustic features are ready</h3>
              <p className="muted-copy">
                These measurements describe how the voice behaves acoustically. They are inputs for the emotion model, not a direct reading of the speaker's internal state.
              </p>
            </div>
            <span className="result-badge success">{featureResult.features.version.toUpperCase()}</span>
          </div>

          <div className="metric-grid feature-metric-grid">
            <div className="metric-item"><span>Mean pitch</span><strong>{featureResult.features.pitch_hz.mean.toFixed(1)} Hz</strong></div>
            <div className="metric-item"><span>Pitch spread</span><strong>{featureResult.features.pitch_hz.std.toFixed(1)} Hz</strong></div>
            <div className="metric-item"><span>Energy</span><strong>{featureResult.features.energy.mean.toFixed(4)}</strong></div>
            <div className="metric-item"><span>Centroid</span><strong>{featureResult.features.spectral_centroid_hz.mean.toFixed(0)} Hz</strong></div>
            <div className="metric-item"><span>Speech activity</span><strong>{(100 * (1 - featureResult.features.silence_ratio)).toFixed(0)}%</strong></div>
            <div className="metric-item"><span>Syllable rate</span><strong>{featureResult.features.estimated_syllable_rate_sps.toFixed(2)} /s</strong></div>
          </div>

          <div className="feature-details">
            <div className="feature-section">
              <div className="feature-section-heading">
                <span>MFCC MEANS</span>
                <small>13 coefficients</small>
              </div>
              <div className="feature-bars">
                {featureResult.features.mfcc_mean.map((value, index) => {
                  const maxAbs = Math.max(...featureResult.features.mfcc_mean.map((item) => Math.abs(item)), 1);
                  const width = Math.min(100, (Math.abs(value) / maxAbs) * 100);
                  return (
                    <div className="feature-bar-row" key={`mfcc-${index}`}>
                      <span>c{index}</span>
                      <div className="feature-bar-track"><i style={{ width: `${width}%` }} /></div>
                      <strong>{value.toFixed(2)}</strong>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="feature-section">
              <div className="feature-section-heading">
                <span>CHROMA PROFILE</span>
                <small>pitch-class energy</small>
              </div>
              <div className="chroma-grid">
                {featureResult.features.chroma.map((value, index) => (
                  <div className="chroma-item" key={`chroma-${index}`}>
                    <span>{["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][index]}</span>
                    <div className="chroma-track"><i style={{ height: `${Math.min(100, value * 100 * 3)}%` }} /></div>
                    <small>{value.toFixed(3)}</small>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <p className="next-step">{featureResult.next_step}</p>
        </div>
      )}



      {fusionError && (
        <div className="message model-pending-message fusion-pending-message" role="status">
          <span className="eyebrow">PHASE 5 · FUSION STATUS</span>
          <strong>Calibrated fusion is not active</strong>
          <span>{fusionError}</span>
        </div>
      )}

      {fusionResult && (
        <div className="emotion-result fusion-result">
          <div className="result-heading">
            <div>
              <p className="eyebrow">PHASE 5 · CALIBRATED HYBRID INFERENCE</p>
              <h3>CEREBRO final emotion estimate</h3>
              <p className="muted-copy">
                The classical acoustic model and the deep speech model are calibrated separately, then combined with learned validation-set weights. Agreement is reported explicitly instead of hidden behind a single score.
              </p>
            </div>
            <span className="result-badge success">FUSION ONLINE</span>
          </div>

          <div className="fusion-summary-grid">
            <div className="emotion-primary fusion-primary">
              <span>FINAL PREDICTION</span>
              <strong>{fusionResult.emotion.toUpperCase()}</strong>
              <small>{(fusionResult.confidence * 100).toFixed(1)}% calibrated confidence</small>
            </div>
            <div className="fusion-agreement-card">
              <div className="fusion-agreement-top">
                <span>MODEL AGREEMENT</span>
                <strong className={`agreement-${fusionResult.agreement}`}>{fusionResult.agreement.toUpperCase()}</strong>
              </div>
              <div className="metric-item"><span>Jensen–Shannon divergence</span><strong>{fusionResult.js_divergence.toFixed(4)}</strong></div>
              <div className="metric-item"><span>Classical weight</span><strong>{(fusionResult.classical.weight * 100).toFixed(0)}%</strong></div>
              <div className="metric-item"><span>Deep weight</span><strong>{(fusionResult.deep.weight * 100).toFixed(0)}%</strong></div>
            </div>
          </div>

          <div className="emotion-probabilities fusion-probabilities">
            {fusionResult.probabilities.map((item) => (
              <div className="emotion-row" key={`fusion-${item.emotion}`}>
                <span>{item.emotion}</span>
                <div className="emotion-track"><i style={{ width: `${Math.max(2, item.probability * 100)}%` }} /></div>
                <strong>{(item.probability * 100).toFixed(1)}%</strong>
              </div>
            ))}
          </div>

          <div className="fusion-model-grid">
            <div className="fusion-model-card">
              <span className="eyebrow">CLASSICAL ACOUSTIC</span>
              <strong>{fusionResult.classical.emotion.toUpperCase()}</strong>
              <small>Raw {(fusionResult.classical.confidence * 100).toFixed(1)}% · calibrated {(fusionResult.classical.calibrated_confidence * 100).toFixed(1)}% · weight {(fusionResult.classical.weight * 100).toFixed(0)}%</small>
            </div>
            <div className="fusion-model-card">
              <span className="eyebrow">DEEP SPEECH</span>
              <strong>{fusionResult.deep.emotion.toUpperCase()}</strong>
              <small>Raw {(fusionResult.deep.confidence * 100).toFixed(1)}% · calibrated {(fusionResult.deep.calibrated_confidence * 100).toFixed(1)}% · weight {(fusionResult.deep.weight * 100).toFixed(0)}%</small>
            </div>
          </div>

          <p className="model-note">Calibration: {fusionResult.calibration_version}. Fusion: {fusionResult.fusion_strategy}. This output estimates emotion conveyed by the speech, not the speaker's internal emotional state.</p>
        </div>
      )}

      {insightsResult && <EmotionInsights result={insightsResult} profileId={profileId} sessionId={recording?.sessionId} />}
      <AffectSpacePanel result={affectResult} error={affectError} />
      <AffectDynamicsPanel result={affectDynamicsResult} error={affectDynamicsError} />

      {transcriptionError && (
        <div className="message model-pending-message" role="status">
          <span className="eyebrow">PHASE 10 · TRANSCRIPTION STATUS</span>
          <strong>Local content transcription is unavailable</strong>
          <span>{transcriptionError}</span>
        </div>
      )}

      {transcriptionResult && (
        <div className="transcription-result-panel">
          <div className="result-heading">
            <div>
              <p className="eyebrow">PHASE 10 · SPEECH CONTENT</p>
              <h3>What CEREBRO heard</h3>
              <p className="muted-copy">
                This transcript is generated from the audio with local Whisper ASR. It provides linguistic context; the emotion system still bases its primary prediction on the spoken acoustic signal.
              </p>
            </div>
            <span className="result-badge success">LOCAL ASR</span>
          </div>
          <div className="transcript-card primary-transcript-card">
            <span>TRANSCRIPT</span>
            <p>{transcriptionResult.transcription.text || "[no speech detected]"}</p>
          </div>
          <div className="metric-grid feature-metric-grid">
            <div className="metric-item"><span>Model</span><strong>{transcriptionResult.transcription.model_id}</strong></div>
            <div className="metric-item"><span>Language</span><strong>{transcriptionResult.transcription.language || "auto"}</strong></div>
            <div className="metric-item"><span>Transcript duration</span><strong>{transcriptionResult.transcription.duration_seconds.toFixed(2)} s</strong></div>
          </div>
        </div>
      )}

      {temporalError && (
        <div className="message model-pending-message" role="status">
          <span className="eyebrow">PHASE 6 · TEMPORAL STATUS</span>
          <strong>Temporal trajectory is not active</strong>
          <span>{temporalError}</span>
        </div>
      )}

      {temporalResult && <TemporalEmotionTimeline result={temporalResult} />}

      <div className="phase-note">
        <span>CURRENT ANALYSIS CONTRACT</span>
        <strong>Audio → acoustic/deep models → calibrated fusion → dimensional affect → affect dynamics → temporal emotion → evidence + uncertainty → personal acoustic baseline → content verification → comparison + robustness</strong>
      </div>
    </section>
  );
}
