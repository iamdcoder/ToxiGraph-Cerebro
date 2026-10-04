function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function signed(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`;
}

function formatSeconds(value) {
  return `${Number(value || 0).toFixed(1)}s`;
}

function titleCase(value) {
  return String(value || "").replace(/\b\w/g, (letter) => letter.toUpperCase()).replaceAll("_", " ");
}

function stateClass(value) {
  return `conversation-state-badge ${String(value || "insufficient_data").replaceAll("_", "-")}`;
}

function stateTone(value) {
  if (["escalating", "peak_tension", "tense"].includes(value)) return "state-hot";
  if (["cooling", "deescalating"].includes(value)) return "state-cooling";
  if (value === "engaged") return "state-engaged";
  return "state-neutral";
}

export default function ConversationStatePanel({ state }) {
  if (!state) return null;

  const points = state.points || [];
  const transitions = state.transitions || [];
  const distribution = Object.entries(state.state_distribution || {}).sort((a, b) => b[1] - a[1]);
  const occupancy = Object.entries(state.duration_by_state || {}).sort((a, b) => b[1] - a[1]);

  return (
    <section className="conversation-state-card">
      <div className="section-header-row compact">
        <div>
          <p className="eyebrow">CONVERSATION STATE ENGINE</p>
          <h3>From measurements to an interpretable conversation state</h3>
          <p className="muted-copy">
            The state machine summarizes observed interaction dynamics. It uses hysteresis so isolated noisy turns do not cause the state to flicker.
          </p>
        </div>
        <div className={`${stateClass(state.end_state)} ${stateTone(state.end_state)}`}>
          {titleCase(state.end_state)}
        </div>
      </div>

      {state.state_count === 0 ? (
        <div className="conversation-state-empty">
          <strong>Insufficient cross-speaker evidence</strong>
          <span>State tracking begins when the recording contains usable speaker-to-speaker response boundaries.</span>
        </div>
      ) : (
        <>
          <div className="conversation-state-metrics">
            <div className="state-metric-card"><span>Start state</span><strong>{titleCase(state.start_state)}</strong></div>
            <div className="state-metric-card"><span>End state</span><strong>{titleCase(state.end_state)}</strong></div>
            <div className="state-metric-card"><span>Peak tension</span><strong>{percent(state.peak_tension)}</strong><small>{state.peak_tension_at_seconds == null ? "—" : `at ${formatSeconds(state.peak_tension_at_seconds)}`}</small></div>
            <div className="state-metric-card"><span>State transitions</span><strong>{state.transition_count}</strong></div>
          </div>

          <div className="state-flow-strip">
            {points.map((point, index) => (
              <div className={`state-flow-node ${stateTone(point.state)}`} key={`${point.index}-${point.at_seconds}`}>
                <span>{formatSeconds(point.at_seconds)}</span>
                <strong>{titleCase(point.state)}</strong>
                {index < points.length - 1 && <i>→</i>}
              </div>
            ))}
          </div>

          <div className="state-detail-grid">
            <div className="state-detail-card">
              <p className="eyebrow">STATE OCCUPANCY</p>
              {occupancy.map(([name, seconds]) => (
                <div className="state-bar-row" key={name}>
                  <div><span>{titleCase(name)}</span><strong>{formatSeconds(seconds)}</strong></div>
                  <div className="state-bar-track"><span style={{ width: `${Math.max(4, Number(state.state_distribution?.[name] || 0) * 100)}%` }} /></div>
                </div>
              ))}
            </div>

            <div className="state-detail-card">
              <p className="eyebrow">STATE TRANSITIONS</p>
              {transitions.length === 0 ? (
                <p className="muted-copy">No state transition crossed the configured hysteresis boundaries.</p>
              ) : (
                <div className="state-transition-list">
                  {transitions.map((item) => (
                    <article className="state-transition-row" key={`${item.index}-${item.at_seconds}`}>
                      <span>{formatSeconds(item.at_seconds)}</span>
                      <strong>{titleCase(item.from_state)} → {titleCase(item.to_state)}</strong>
                      <em>{item.trigger.replaceAll("_", " ")}</em>
                      <small>Δ tension {signed(item.tension_change)}</small>
                    </article>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="state-timeline-card">
            <div className="section-header-row compact">
              <div>
                <p className="eyebrow">STATE TIMELINE</p>
                <h4>Observed response boundary → state estimate</h4>
              </div>
              <span className="muted-copy">{points.length} state points</span>
            </div>
            <div className="state-timeline-list">
              {points.map((point) => (
                <article className="state-timeline-row" key={`${point.index}-${point.at_seconds}`}>
                  <span className="state-timeline-time">{formatSeconds(point.at_seconds)}</span>
                  <div>
                    <strong>{point.from_speaker} → {point.to_speaker}</strong>
                    <span>{point.emotion_transition}</span>
                  </div>
                  <span className={`${stateClass(point.state)} ${stateTone(point.state)}`}>{titleCase(point.state)}</span>
                  <span className="state-tension-number">T {percent(point.tension_score)} · Δ {signed(point.directional_signal)}</span>
                </article>
              ))}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
