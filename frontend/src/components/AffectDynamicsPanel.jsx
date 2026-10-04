const QUADRANT_SHORT = {
  "positive / activated": "POSITIVE · ACTIVATED",
  "positive / calm": "POSITIVE · CALM",
  "negative / activated": "NEGATIVE · ACTIVATED",
  "negative / calm": "NEGATIVE · CALM",
  "mixed / transitional": "MIXED · TRANSITIONAL",
};

function pct(value) {
  return `${(value * 100).toFixed(0)}%`;
}

function time(seconds) {
  return `${seconds.toFixed(1)}s`;
}

function signed(value) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(2)}`;
}

function pointColor(quadrant) {
  if (quadrant === "negative / activated") return "#ff6b7d";
  if (quadrant === "negative / calm") return "#ff9a62";
  if (quadrant === "positive / activated") return "#79e2bd";
  if (quadrant === "positive / calm") return "#6cb8ff";
  return "#c6b4ff";
}

function trajectoryPath(points, width = 640, height = 300) {
  const speech = points.filter((point) => point.is_speech && point.smoothed);
  if (speech.length < 2) return null;
  return speech.map((point, index) => {
    const x = 28 + point.smoothed.valence * (width - 56);
    const y = height - 28 - point.smoothed.arousal * (height - 56);
    return `${index === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)}`;
  }).join(" ");
}

export default function AffectDynamicsPanel({ result, error = "" }) {
  if (error) {
    return (
      <section className="affect-dynamics panel-card">
        <div className="result-heading">
          <div>
            <p className="eyebrow">PHASE 15 · AFFECT DYNAMICS</p>
            <h3>Affect trajectory unavailable</h3>
          </div>
          <span className="result-badge warning">NOT ACTIVE</span>
        </div>
        <div className="message model-pending-message">
          {error}
        </div>
      </section>
    );
  }

  if (!result) return null;

  const path = trajectoryPath(result.points);
  const events = result.events || [];
  const state = result.end_state;

  return (
    <section className="affect-dynamics panel-card">
      <div className="result-heading">
        <div>
          <p className="eyebrow">PHASE 15 · AFFECT DYNAMICS</p>
          <h3>How the emotional state moved through the speech</h3>
          <p className="muted-copy">
            CEREBRO tracks continuous valence, arousal, and dominance over overlapping windows.
            The path describes changes in the speech signal; it is not a measurement of private internal emotion.
          </p>
        </div>
        <span className="result-badge success">TRAJECTORY ONLINE</span>
      </div>

      <div className="affect-dynamics-grid">
        <div className="affect-dynamics-map-wrap">
          <div className="affect-dynamics-axis-label left">LOW AROUSAL</div>
          <div className="affect-dynamics-axis-label right">HIGH AROUSAL</div>
          <div className="affect-dynamics-map">
            <span className="affect-dynamics-label top">POSITIVE VALENCE</span>
            <span className="affect-dynamics-label bottom">NEGATIVE VALENCE</span>
            <div className="affect-dynamics-cross horizontal" />
            <div className="affect-dynamics-cross vertical" />
            {path && <svg viewBox="0 0 640 300" preserveAspectRatio="none" aria-label="Valence arousal trajectory"><path d={path} fill="none" stroke="rgba(198,180,255,.92)" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" /></svg>}
            {result.points.filter((point) => point.is_speech && point.smoothed).map((point) => {
              const left = 6 + point.smoothed.valence * 88;
              const top = 94 - point.smoothed.arousal * 88;
              return (
                <span
                  className="affect-dynamics-point"
                  key={`affect-point-${point.index}`}
                  title={`${time(point.start_seconds)} · ${QUADRANT_SHORT[point.quadrant] || point.quadrant || "TRANSITION"}`}
                  style={{ left: `${left}%`, top: `${top}%`, background: pointColor(point.quadrant) }}
                />
              );
            })}
          </div>
        </div>

        <div className="affect-dynamics-summary">
          <div className="metric-item"><span>Dominant region</span><strong>{QUADRANT_SHORT[result.dominant_quadrant] || result.dominant_quadrant || "Not available"}</strong></div>
          <div className="metric-item"><span>Trajectory stability</span><strong>{pct(result.stability_score)}</strong></div>
          <div className="metric-item"><span>Speech coverage</span><strong>{pct(result.speech_coverage)}</strong></div>
          <div className="metric-item"><span>Peak arousal</span><strong>{result.peak_arousal == null ? "—" : `${pct(result.peak_arousal)} @ ${time(result.peak_arousal_at_seconds)}`}</strong></div>
          <div className="metric-item"><span>Lowest valence</span><strong>{result.lowest_valence == null ? "—" : `${pct(result.lowest_valence)} @ ${time(result.lowest_valence_at_seconds)}`}</strong></div>
          <div className="metric-item"><span>Largest affect shift</span><strong>{result.largest_shift.toFixed(3)}</strong></div>
        </div>
      </div>

      <div className="affect-delta-grid">
        {Object.entries(result.net_change).map(([dimension, value]) => (
          <div className="metric-item" key={dimension}>
            <span>Net {dimension}</span>
            <strong>{signed(value)}</strong>
          </div>
        ))}
        <div className="metric-item"><span>Mean velocity</span><strong>{result.mean_velocity.toFixed(3)} /s</strong></div>
        <div className="metric-item"><span>Peak velocity</span><strong>{result.peak_velocity.toFixed(3)} /s</strong></div>
      </div>

      <div className="affect-event-panel">
        <div className="feature-section-heading">
          <span>DETECTED AFFECT EVENTS</span>
          <small>{events.length} event{events.length === 1 ? "" : "s"}</small>
        </div>
        {events.length === 0 ? (
          <p className="muted-copy">No threshold-crossing affect events were detected.</p>
        ) : (
          <div className="affect-event-list">
            {events.slice(0, 12).map((event, index) => (
              <div className="affect-event-card" key={`affect-event-${index}`}>
                <div>
                  <span className="eyebrow">{event.event_type.replaceAll("_", " ").toUpperCase()}</span>
                  <strong>{event.dimension ? event.dimension.toUpperCase() : "WHOLE AFFECT STATE"}</strong>
                </div>
                <div className="affect-event-meta">
                  <span>{time(event.at_seconds)}</span>
                  <span>score {event.score.toFixed(2)}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {state && (
        <div className="affect-current-state">
          <div>
            <span className="eyebrow">ENDING STATE</span>
            <strong>{pct(state.valence)} V · {pct(state.arousal)} A · {pct(state.dominance)} D</strong>
          </div>
          <small>{events.length ? "The trajectory contains measurable affect transitions." : "The trajectory remained comparatively stable."}</small>
        </div>
      )}
    </section>
  );
}
