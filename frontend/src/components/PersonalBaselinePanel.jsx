import { useEffect, useMemo, useState } from "react";

import {
  createPersonalBaseline,
  deletePersonalBaseline,
  getPersonalBaselineStatus,
} from "../api";

const PROFILE_STORAGE_KEY = "cerebro-profile-id";

function getProfileId() {
  try {
    const existing = window.localStorage.getItem(PROFILE_STORAGE_KEY);
    if (existing) return existing;
    const generated = typeof window.crypto?.randomUUID === "function"
      ? window.crypto.randomUUID()
      : `profile-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    window.localStorage.setItem(PROFILE_STORAGE_KEY, generated);
    return generated;
  } catch {
    return `profile-${Date.now()}`;
  }
}
const MIN_SAMPLES = 3;
const MAX_SAMPLES = 8;

export default function PersonalBaselinePanel({ recording, onProfileChange }) {
  const [samples, setSamples] = useState([]);
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const [profileId] = useState(getProfileId);

  useEffect(() => {
    let mounted = true;
    getPersonalBaselineStatus(profileId)
      .then((payload) => {
        if (!mounted) return;
        setStatus(payload);
        onProfileChange(profileId);
      })
      .catch(() => {
        if (!mounted) return;
        setStatus({ available: false, profile_id: profileId, sample_count: 0 });
        onProfileChange(profileId);
      });
    return () => {
      mounted = false;
    };
  }, [onProfileChange, profileId]);

  const sampleCount = samples.length;
  const canBuild = sampleCount >= MIN_SAMPLES && sampleCount <= MAX_SAMPLES;
  const progressWidth = useMemo(
    () => `${Math.min(100, (sampleCount / MIN_SAMPLES) * 100)}%`,
    [sampleCount],
  );

  function addCurrentRecording() {
    if (!recording?.blob) return;
    if (samples.length >= MAX_SAMPLES) return;
    setSamples((current) => [...current, recording.blob]);
    setMessage(`Reference sample ${sampleCount + 1} added. Record different natural speech for the strongest baseline.`);
    setError("");
  }

  function removeSample(index) {
    setSamples((current) => current.filter((_, itemIndex) => itemIndex !== index));
    setMessage("Reference sample removed.");
  }

  async function buildBaseline() {
    if (!canBuild) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const payload = await createPersonalBaseline(profileId, samples);
      setStatus(payload.profile);
      setSamples([]);
      onProfileChange(profileId);
      setMessage("Personal baseline is ready. New analyses will now report deviations from this speaker's own acoustic profile.");
    } catch (err) {
      setError(err?.message || "The personal baseline could not be created.");
    } finally {
      setBusy(false);
    }
  }

  async function resetBaseline() {
    setBusy(true);
    setError("");
    try {
      await deletePersonalBaseline(profileId);
      setStatus({ available: false, profile_id: profileId, sample_count: 0 });
      onProfileChange(profileId);
      setMessage("Personal baseline deleted.");
    } catch (err) {
      setError(err?.message || "The personal baseline could not be deleted.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="baseline-panel">
      <div className="result-heading">
        <div>
          <p className="eyebrow">PHASE 8 · PERSONAL VOICE BASELINE</p>
          <h3>Learn what this voice normally sounds like</h3>
          <p className="muted-copy">
            CEREBRO stores an acoustic reference profile from your natural speech. It does not retrain the emotion model; it only adds speaker-specific context to later analysis.
          </p>
        </div>
        <span className={`result-badge ${status?.available ? "success" : "warning"}`}>
          {status?.available ? "BASELINE READY" : "NOT SET"}
        </span>
      </div>

      <div className="baseline-grid">
        <div className="baseline-status-card">
          <div className="feature-section-heading">
            <span>REFERENCE SAMPLES</span>
            <small>{sampleCount}/{MAX_SAMPLES} queued</small>
          </div>
          <div className="baseline-progress"><i style={{ width: progressWidth }} /></div>
          <div className="baseline-samples">
            {samples.map((_, index) => (
              <button
                type="button"
                className="baseline-sample"
                key={`baseline-sample-${index}`}
                onClick={() => removeSample(index)}
                title="Remove this reference sample"
              >
                SAMPLE {index + 1}
                <span>×</span>
              </button>
            ))}
            {!samples.length && <span className="baseline-empty">Record at least 3 natural samples, then add each one here.</span>}
          </div>

          <div className="baseline-actions">
            <button
              className="secondary-button"
              type="button"
              onClick={addCurrentRecording}
              disabled={!recording?.blob || sampleCount >= MAX_SAMPLES || busy}
            >
              Add latest recording
            </button>
            <button
              className="primary-button"
              type="button"
              onClick={buildBaseline}
              disabled={!canBuild || busy}
            >
              {busy ? "Working…" : "Build personal baseline"}
            </button>
          </div>
        </div>

        <div className="baseline-profile-card">
          <span className="eyebrow">ACTIVE PROFILE</span>
          {status?.available ? (
            <>
              <strong>{status.profile_id.toUpperCase()}</strong>
              <span>{status.sample_count} reference recordings · {status.reference_duration_seconds.toFixed(1)} s average sample duration</span>
              <button className="text-button" type="button" onClick={resetBaseline} disabled={busy}>Reset profile</button>
            </>
          ) : (
            <>
              <strong>THIS BROWSER</strong>
              <span>No personal baseline is active yet.</span>
              <small>Use varied, natural speech instead of intentionally emotional readings for the reference samples.</small>
            </>
          )}
        </div>
      </div>

      {message && <p className="baseline-message">{message}</p>}
      {error && <p className="baseline-error" role="alert">{error}</p>}
    </div>
  );
}
