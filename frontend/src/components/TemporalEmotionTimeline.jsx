const EMOTION_ICONS = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😊",
  neutral: "😐",
  sad: "😢",
  surprised: "😮",
};

function formatSeconds(value) {
  const total = Math.max(0, value || 0);
  const minutes = Math.floor(total / 60).toString().padStart(2, "0");
  const seconds = Math.floor(total % 60).toString().padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function emotionLabel(value) {
  if (!value) return "silence";
  return value.replaceAll("_", " ");
}

export default function TemporalEmotionTimeline({ result }) {
  if (!result) return null;

  return (
    <div className="temporal-result">
      <div className="result-heading">
        <div>
          <p className="eyebrow">PHASE 6 · TEMPORAL EMOTION ANALYSIS</p>
          <h3>Emotion through time</h3>
          <p className="muted-copy">
            CEREBRO analyzes overlapping three-second windows and smooths their probability distributions. Silence is explicitly excluded from emotion labeling.
          </p>
        </div>
        <span className="result-badge success">TRAJECTORY READY</span>
      </div>

      <div className="temporal-summary-grid">
        <div className="metric-item">
          <span>Dominant emotion</span>
          <strong>{EMOTION_ICONS[result.dominant_emotion] || "•"} {emotionLabel(result.dominant_emotion)}</strong>
        </div>
        <div className="metric-item">
          <span>Speech coverage</span>
          <strong>{(result.speech_coverage * 100).toFixed(0)}%</strong>
        </div>
        <div className="metric-item">
          <span>Windows analyzed</span>
          <strong>{result.speech_windows} / {result.analyzed_windows}</strong>
        </div>
        <div className="metric-item">
          <span>Transitions detected</span>
          <strong>{result.transitions.length}</strong>
        </div>
      </div>

      <div className="temporal-track-wrap">
        <div className="temporal-axis">
          <span>0:00</span>
          <span>{formatSeconds(result.duration_seconds)}</span>
        </div>
        <div className="temporal-track" role="img" aria-label="Emotion trajectory over time">
          {result.segments.map((segment) => {
            const width = Math.max(
              1,
              ((segment.end_seconds - segment.start_seconds) / Math.max(result.duration_seconds, 0.001)) * 100,
            );
            const emotionClass = segment.is_speech ? `emotion-${segment.emotion}` : "emotion-silence";
            return (
              <div
                className={`temporal-segment ${emotionClass} ${segment.transition ? "has-transition" : ""}`}
                key={segment.index}
                style={{ width: `${width}%` }}
                title={`${formatSeconds(segment.start_seconds)}–${formatSeconds(segment.end_seconds)} · ${emotionLabel(segment.emotion)} · ${(segment.confidence * 100).toFixed(1)}%`}
              >
                <span>{segment.is_speech ? EMOTION_ICONS[segment.emotion] || "•" : "∅"}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="temporal-segment-list">
        {result.segments.map((segment) => (
          <div className={`temporal-segment-card ${segment.transition ? "transition-card" : ""}`} key={`card-${segment.index}`}>
            <div className="temporal-card-time">{formatSeconds(segment.start_seconds)} → {formatSeconds(segment.end_seconds)}</div>
            <div className="temporal-card-main">
              <span className="temporal-emoji">{segment.is_speech ? EMOTION_ICONS[segment.emotion] || "•" : "∅"}</span>
              <div>
                <strong>{emotionLabel(segment.emotion)}</strong>
                <small>
                  {segment.is_speech
                    ? `${(segment.confidence * 100).toFixed(1)}% smoothed confidence · ${(segment.speech_ratio * 100).toFixed(0)}% speech`
                    : `${(segment.speech_ratio * 100).toFixed(0)}% speech · excluded from emotion inference`}
                </small>
              </div>
            </div>
            {segment.transition && (
              <span className="temporal-transition-badge">TRANSITION · {segment.transition_score.toFixed(3)}</span>
            )}
          </div>
        ))}
      </div>

      {result.transitions.length > 0 && (
        <div className="temporal-transitions">
          <div className="feature-section-heading">
            <span>DETECTED TRANSITIONS</span>
            <small>{result.transitions.length} transition(s)</small>
          </div>
          <div className="transition-list">
            {result.transitions.map((transition, index) => (
              <div className="transition-row" key={`${transition.source_segment_index}-${index}`}>
                <span>{formatSeconds(transition.at_seconds)}</span>
                <strong>{EMOTION_ICONS[transition.from_emotion] || "•"} {emotionLabel(transition.from_emotion)}</strong>
                <span>→</span>
                <strong>{EMOTION_ICONS[transition.to_emotion] || "•"} {emotionLabel(transition.to_emotion)}</strong>
                <small>{transition.score.toFixed(3)}</small>
              </div>
            ))}
          </div>
        </div>
      )}

      <p className="model-note">
        Windowing: {result.window_seconds.toFixed(1)} s with {result.hop_seconds.toFixed(1)} s stride · smoothing α={result.smoothing_alpha.toFixed(2)} · calibration {result.calibration_version}.
      </p>
    </div>
  );
}
