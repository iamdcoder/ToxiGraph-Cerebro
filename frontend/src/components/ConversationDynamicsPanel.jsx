function formatNumber(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  return Number(value).toFixed(digits);
}

function percent(value) {
  return `${Math.round(Number(value || 0) * 100)}%`;
}

function signed(value, digits = 2) {
  if (value == null || Number.isNaN(Number(value))) return "—";
  const number = Number(value);
  return `${number >= 0 ? "+" : ""}${number.toFixed(digits)}`;
}

function titleCase(value) {
  return String(value || "").replace(/\b\w/g, (letter) => letter.toUpperCase()).replaceAll("_", " ");
}

function patternClass(pattern) {
  return `conversation-dynamics-pattern ${String(pattern || "insufficient_data").replaceAll("_", "-")}`;
}

function eventTone(type) {
  if (type === "escalation") return "dynamics-event-escalation";
  if (type === "deescalation") return "dynamics-event-deescalation";
  if (type === "mixed_shift") return "dynamics-event-mixed";
  return "dynamics-event-stable";
}

export default function ConversationDynamicsPanel({ dynamics }) {
  if (!dynamics) return null;

  const events = dynamics.events || [];
  const pairs = dynamics.pair_metrics || [];
  const trend = Number(dynamics.tension_trend || 0);
  const tension = Number(dynamics.net_tension || 0);
  const tensionFill = `${Math.max(0, Math.min(100, 50 + tension * 50))}%`;

  return (
    <section className="conversation-dynamics-card">
      <div className="section-header-row compact">
        <div>
          <p className="eyebrow">CONVERSATION EMOTION DYNAMICS</p>
          <h3>How the interaction moved emotionally</h3>
          <p className="muted-copy">
            CEREBRO measures observed turn-to-turn changes across speakers. These signals describe the interaction sequence; they do not establish causation or private emotional states.
          </p>
        </div>
        <div className={patternClass(dynamics.dominant_pattern)}>{titleCase(dynamics.dominant_pattern)}</div>
      </div>

      {dynamics.event_count === 0 ? (
        <div className="conversation-dynamics-empty">
          <strong>Not enough cross-speaker turns</strong>
          <span>At least two speakers with alternating analyzed turns are needed for conversation dynamics.</span>
        </div>
      ) : (
        <>
          <div className="conversation-dynamics-metrics">
            <div className="dynamics-metric-card">
              <span>Net tension</span>
              <strong>{signed(tension)}</strong>
              <small>−1 de-escalating · +1 escalating</small>
            </div>
            <div className="dynamics-metric-card">
              <span>Affect synchrony</span>
              <strong>{dynamics.affect_synchrony == null ? "—" : percent(dynamics.affect_synchrony)}</strong>
              <small>State similarity at response boundaries</small>
            </div>
            <div className="dynamics-metric-card">
              <span>Volatility</span>
              <strong>{percent(dynamics.volatility)}</strong>
              <small>Mean normalized V/A/D movement</small>
            </div>
            <div className="dynamics-metric-card">
              <span>Response latency</span>
              <strong>{dynamics.median_response_gap_seconds == null ? "—" : `${formatNumber(dynamics.median_response_gap_seconds, 1)}s`}</strong>
              <small>Median cross-speaker gap</small>
            </div>
          </div>

          <div className="conversation-tension-strip">
            <div className="conversation-tension-track">
              <span className="conversation-tension-mid" />
              <span className="conversation-tension-marker" style={{ left: tensionFill }} />
            </div>
            <div className="conversation-tension-labels">
              <span>DE-ESCALATING</span>
              <span>NEUTRAL</span>
              <span>ESCALATING</span>
            </div>
          </div>

          <div className="dynamics-detail-grid">
            <div className="dynamics-detail-card">
              <p className="eyebrow">TRAJECTORY</p>
              <div className="dynamics-detail-row"><span>Escalation score</span><strong>{percent(dynamics.escalation_score)}</strong></div>
              <div className="dynamics-detail-row"><span>De-escalation score</span><strong>{percent(dynamics.deescalation_score)}</strong></div>
              <div className="dynamics-detail-row"><span>Tension trend</span><strong>{signed(trend)}</strong></div>
              <div className="dynamics-detail-row"><span>Stability</span><strong>{percent(dynamics.stability_score)}</strong></div>
            </div>
            <div className="dynamics-detail-card">
              <p className="eyebrow">RESPONSE BEHAVIOR</p>
              <div className="dynamics-detail-row"><span>Fast responses</span><strong>{percent(dynamics.fast_response_share)}</strong></div>
              <div className="dynamics-detail-row"><span>Mean response gap</span><strong>{dynamics.mean_response_gap_seconds == null ? "—" : `${formatNumber(dynamics.mean_response_gap_seconds, 1)}s`}</strong></div>
              <div className="dynamics-detail-row"><span>90th percentile gap</span><strong>{dynamics.p90_response_gap_seconds == null ? "—" : `${formatNumber(dynamics.p90_response_gap_seconds, 1)}s`}</strong></div>
              <div className="dynamics-detail-row"><span>Latency shift</span><strong>{dynamics.response_latency_shift_seconds == null ? "—" : `${signed(dynamics.response_latency_shift_seconds, 1)}s`}</strong></div>
            </div>
            <div className="dynamics-detail-card">
              <p className="eyebrow">SHIFT COUNTS</p>
              <div className="dynamics-detail-row"><span>Arousal rises</span><strong>{dynamics.arousal_rise_events}</strong></div>
              <div className="dynamics-detail-row"><span>Valence drops</span><strong>{dynamics.valence_drop_events}</strong></div>
              <div className="dynamics-detail-row"><span>Escalation events</span><strong>{dynamics.escalation_events}</strong></div>
              <div className="dynamics-detail-row"><span>De-escalation events</span><strong>{dynamics.deescalation_events}</strong></div>
            </div>
          </div>

          {pairs.length > 0 && (
            <div className="dynamics-pairs-card">
              <div className="section-header-row compact">
                <div>
                  <p className="eyebrow">PAIR DYNAMICS</p>
                  <h4>Observed response behavior by direction</h4>
                </div>
              </div>
              <div className="dynamics-pair-list">
                {pairs.map((pair) => (
                  <article className="dynamics-pair-card" key={`${pair.from_speaker}-${pair.to_speaker}`}>
                    <div className="dynamics-pair-heading">
                      <strong>{pair.from_speaker} → {pair.to_speaker}</strong>
                      <span>{pair.response_count} response{pair.response_count === 1 ? "" : "s"}</span>
                    </div>
                    <div className="dynamics-pair-stats">
                      <span>escalation {pair.escalation_count}</span>
                      <span>de-escalation {pair.deescalation_count}</span>
                      <span>median gap {formatNumber(pair.median_gap_seconds, 1)}s</span>
                      <span>synchrony {pair.affect_synchrony == null ? "—" : percent(pair.affect_synchrony)}</span>
                    </div>
                    <div className="dynamics-pair-affect">
                      <span>ΔV {signed(pair.mean_valence_change)}</span>
                      <span>ΔA {signed(pair.mean_arousal_change)}</span>
                      <span>movement {percent(pair.mean_affect_magnitude)}</span>
                    </div>
                  </article>
                ))}
              </div>
            </div>
          )}

          <div className="dynamics-events-card">
            <div className="section-header-row compact">
              <div>
                <p className="eyebrow">TURN DYNAMICS</p>
                <h4>Cross-speaker affect events</h4>
              </div>
              <span className="muted-copy">{events.length} observed response boundaries</span>
            </div>
            <div className="dynamics-event-list">
              {events.map((event) => (
                <article className={`dynamics-event-row ${eventTone(event.event_type)}`} key={`${event.index}-${event.at_seconds}`}>
                  <span className="dynamics-event-time">{formatNumber(event.at_seconds, 1)}s</span>
                  <div>
                    <strong>{event.from_speaker} → {event.to_speaker}</strong>
                    <span>{titleCase(event.previous_emotion)} → {titleCase(event.next_emotion)}</span>
                  </div>
                  <span className="dynamics-event-type">{titleCase(event.event_type)}</span>
                  <span className="dynamics-event-change">ΔV {signed(event.valence_change)} · ΔA {signed(event.arousal_change)}</span>
                </article>
              ))}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
