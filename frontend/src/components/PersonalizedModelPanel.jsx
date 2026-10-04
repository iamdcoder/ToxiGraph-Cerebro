import { useEffect, useMemo, useState } from "react";

import {
  deletePersonalization,
  getPersonalizationStatus,
  predictPersonalizedEmotion,
  submitPersonalizationFeedback,
} from "../api";

const EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"];
const EMOJI = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😊",
  neutral: "😐",
  sad: "😢",
  surprised: "😮",
};

function probabilityMap(result) {
  return Object.fromEntries((result?.fusion?.probabilities || []).map((item) => [item.emotion, item.probability]));
}

export default function PersonalizedModelPanel({ profileId, result, sessionId }) {
  const [status, setStatus] = useState(null);
  const [selectedLabel, setSelectedLabel] = useState(result?.fusion?.emotion || "");
  const [prediction, setPrediction] = useState(result?.insights?.personalization || null);
  const [busy, setBusy] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (!profileId) return undefined;
    let mounted = true;
    getPersonalizationStatus(profileId)
      .then((payload) => {
        if (mounted) setStatus(payload);
      })
      .catch((err) => {
        if (mounted) setError(err?.message || "Personalized calibration status is unavailable.");
      });
    return () => {
      mounted = false;
    };
  }, [profileId]);

  useEffect(() => {
    setSelectedLabel(result?.fusion?.emotion || "");
    setPrediction(result?.insights?.personalization || null);
  }, [result?.fusion?.emotion, result?.insights?.personalization]);

  const probabilities = useMemo(() => probabilityMap(result), [result]);
  const ready = Boolean(status?.model_ready);
  const progress = status ? Math.min(100, (status.feedback_count / status.minimum_feedback) * 100) : 0;

  async function submitFeedback() {
    if (!profileId || !result?.fusion || !selectedLabel || busy) return;
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const acoustic = {
        pitch_mean_hz: result.features.pitch_hz.mean,
        pitch_std_hz: result.features.pitch_hz.std,
        energy_mean: result.features.energy.mean,
        energy_std: result.features.energy.std,
        estimated_syllable_rate_sps: result.features.estimated_syllable_rate_sps,
        estimated_pause_ratio: result.features.estimated_pause_ratio,
      };
      const feedback = await submitPersonalizationFeedback(profileId, {
        session_id: sessionId || `feedback-${Date.now()}`,
        label: selectedLabel,
        quality_score: result.audio_quality?.quality_score ?? null,
        fused_probabilities: probabilities,
        acoustic,
      });
      setStatus(feedback.status);
      const personalized = await predictPersonalizedEmotion(profileId, {
        fused_probabilities: probabilities,
        acoustic,
      });
      setPrediction(personalized.prediction);
      setMessage(
        feedback.status.model_ready
          ? "Speaker-specific calibration is active. The personalized view now uses your confirmed feedback."
          : "Feedback saved. Keep labeling genuinely different conveyed emotions until the calibration gate is reached."
      );
    } catch (err) {
      setError(err?.message || "The feedback could not be saved.");
    } finally {
      setBusy(false);
    }
  }

  async function reset() {
    if (!profileId || resetting) return;
    setResetting(true);
    setError("");
    try {
      await deletePersonalization(profileId);
      const fresh = await getPersonalizationStatus(profileId);
      setStatus(fresh);
      setPrediction(null);
      setMessage("Speaker-specific feedback and calibration were deleted. The global model remains unchanged.");
    } catch (err) {
      setError(err?.message || "Personalized calibration could not be deleted.");
    } finally {
      setResetting(false);
    }
  }

  if (!profileId || !result?.fusion) return null;

  return (
    <div className="personalized-model-card">
      <div className="result-heading">
        <div>
          <p className="eyebrow">PHASE 24 · PERSONALIZED EMOTION CALIBRATION</p>
          <h3>Teach CEREBRO this voice</h3>
          <p className="muted-copy">
            Confirm the emotion the speech was intended to convey. CEREBRO learns a small speaker-specific calibration layer from your explicit labels; the global model and benchmark remain untouched.
          </p>
        </div>
        <span className={`result-badge ${ready ? "success" : "warning"}`}>
          {ready ? "CALIBRATION ACTIVE" : "LEARNING"}
        </span>
      </div>

      <div className="personalization-enrollment">
        <div className="personalization-progress-copy">
          <span>FEEDBACK ENROLLMENT</span>
          <strong>{status ? `${status.feedback_count}/${status.minimum_feedback}` : "—"}</strong>
        </div>
        <div className="baseline-progress"><i style={{ width: `${progress}%` }} /></div>
        <small>
          {status
            ? `${status.unique_labels}/${status.minimum_unique_labels} emotion classes represented. Low-quality feedback is excluded from training.`
            : "Loading speaker-specific calibration status…"}
        </small>
      </div>

      <div className="personalization-feedback-grid">
        <div className="personalization-label-picker">
          <div className="feature-section-heading">
            <span>WHAT EMOTION DID YOU CONVEY?</span>
            <small>explicit speaker feedback</small>
          </div>
          <div className="personalization-emotion-buttons">
            {EMOTIONS.map((emotion) => (
              <button
                key={emotion}
                type="button"
                className={selectedLabel === emotion ? "selected" : ""}
                onClick={() => setSelectedLabel(emotion)}
                disabled={busy}
              >
                <span>{EMOJI[emotion]}</span>
                {emotion}
              </button>
            ))}
          </div>
          <button className="primary-button personalization-submit" type="button" onClick={submitFeedback} disabled={!selectedLabel || busy}>
            {busy ? "Updating calibration…" : `Confirm ${selectedLabel || "emotion"}`}
          </button>
        </div>

        {prediction && (
          <div className="personalization-prediction-card">
            <div className="feature-section-heading">
              <span>PERSONALIZED VIEW</span>
              <small>{prediction.model_ready ? `${prediction.sample_count} labeled sessions` : "global fallback"}</small>
            </div>
            <div className="personalization-prediction-main">
              <span>{EMOJI[prediction.personalized_emotion] || "•"}</span>
              <strong>{prediction.personalized_emotion.toUpperCase()}</strong>
              <small>{(prediction.personalized_confidence * 100).toFixed(1)}% personalized probability</small>
            </div>
            <div className="personalization-compare-row">
              <span>Global</span>
              <strong>{prediction.global_emotion.toUpperCase()} · {(prediction.global_confidence * 100).toFixed(1)}%</strong>
            </div>
            <div className="personalization-compare-row">
              <span>Personalized weight</span>
              <strong>{(prediction.adjustment_weight * 100).toFixed(0)}%</strong>
            </div>
            <div className="emotion-probabilities">
              {EMOTIONS.map((emotion) => {
                const value = prediction.probabilities?.[emotion] || 0;
                return (
                  <div className="emotion-row" key={`personalized-${emotion}`}>
                    <span>{emotion}</span>
                    <div className="emotion-track"><i style={{ width: `${Math.max(2, value * 100)}%` }} /></div>
                    <strong>{(value * 100).toFixed(1)}%</strong>
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {message && <p className="personalization-message">{message}</p>}
      {error && <p className="baseline-error" role="alert">{error}</p>}
      {status?.feedback_count > 0 && (
        <button className="text-button danger-text" type="button" onClick={reset} disabled={resetting || busy}>
          {resetting ? "Deleting…" : "Delete speaker-specific calibration"}
        </button>
      )}
    </div>
  );
}
