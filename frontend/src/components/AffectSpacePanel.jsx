const QUADRANT_LABELS = {
  "positive / activated": "POSITIVE · ACTIVATED",
  "positive / calm": "POSITIVE · CALM",
  "negative / activated": "NEGATIVE · ACTIVATED",
  "negative / calm": "NEGATIVE · CALM",
  "mixed / transitional": "MIXED · TRANSITIONAL",
};

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

function signedValence(value) {
  return ((value * 2) - 1).toFixed(2);
}

function Gauge({ label, value, note }) {
  return (
    <div className="affect-gauge">
      <div className="affect-gauge-head">
        <span>{label}</span>
        <strong>{pct(value)}</strong>
      </div>
      <div className="affect-track">
        <i style={{ width: `${Math.max(2, value * 100)}%` }} />
      </div>
      <small>{note}</small>
    </div>
  );
}

export default function AffectSpacePanel({ result, error = "" }) {
  if (error) {
    return (
      <section className="affect-panel panel-card">
        <div className="section-kicker">PHASE 14 · DIMENSIONAL AFFECT</div>
        <h3>Continuous affect analysis is unavailable.</h3>
        <p className="muted-copy">{error}</p>
      </section>
    );
  }

  if (!result) return null;

  const { values, interpretation } = result;

  return (
    <section className="affect-panel panel-card">
      <div className="result-heading">
        <div>
          <p className="eyebrow">PHASE 14 · CONTINUOUS EMOTIONAL SPACE</p>
          <h3>Beyond a single emotion label</h3>
          <p className="muted-copy">
            CEREBRO estimates where the speech sits in a continuous valence–arousal–dominance space.
          </p>
        </div>
        <span className="result-badge success">DIMENSIONAL MODEL</span>
      </div>

      <div className="affect-hero-grid">
        <div className="affect-space-map">
          <div className="affect-axis-label top">HIGH AROUSAL</div>
          <div className="affect-axis-label bottom">LOW AROUSAL</div>
          <div className="affect-axis-label left">NEGATIVE</div>
          <div className="affect-axis-label right">POSITIVE</div>
          <div
            className="affect-point"
            style={{
              left: `${values.valence * 100}%`,
              bottom: `${values.arousal * 100}%`,
            }}
          />
          <div className="affect-crosshair horizontal" />
          <div className="affect-crosshair vertical" />
        </div>

        <div className="affect-summary-card">
          <span className="affect-quadrant-label">{QUADRANT_LABELS[interpretation.quadrant] || interpretation.quadrant}</span>
          <strong>{interpretation.summary}</strong>
          <small>{result.scale}</small>
        </div>
      </div>

      <div className="affect-gauge-grid">
        <Gauge label="VALENCE" value={values.valence} note={`Signed view: ${signedValence(values.valence)} · ${interpretation.valence_label}`} />
        <Gauge label="AROUSAL" value={values.arousal} note={`Activation: ${interpretation.arousal_label}`} />
        <Gauge label="DOMINANCE" value={values.dominance} note={`Level: ${interpretation.dominance_label}`} />
      </div>

      <div className="affect-footnote">
        <strong>{result.model_name}</strong>
        <span>Continuous affect estimates describe the speech signal; they are not measurements of the speaker's private internal state.</span>
      </div>
    </section>
  );
}
