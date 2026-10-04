import PersonalizedModelPanel from "./PersonalizedModelPanel";

const EMOTION_EMOJI = {
  angry: "😠",
  disgust: "🤢",
  fear: "😨",
  happy: "😊",
  neutral: "😐",
  sad: "😢",
  surprised: "😮",
};

function pct(value) {
  return `${(value * 100).toFixed(1)}%`;
}

export default function EmotionInsights({ result, profileId, sessionId }) {
  if (!result?.insights) return null;
  const { insights } = result;
  const uncertainty = insights.uncertainty;
  const top = result.fusion;

  return (
    <div className="insights-result">
      <div className="result-heading">
        <div>
          <p className="eyebrow">PHASE 7 · EXPLAINABILITY + UNCERTAINTY</p>
          <h3>Why CEREBRO made this prediction</h3>
          <p className="muted-copy">Evidence is separated from interpretation. Measured acoustic signals and model agreement are shown explicitly.</p>
        </div>
        <span className={`result-badge ${uncertainty.level === "low" ? "success" : "warning"}`}>
          {uncertainty.level.replace("_", " ").toUpperCase()} UNCERTAINTY
        </span>
      </div>

      <div className="insights-summary-grid">
        <div className="insight-summary-card">
          <span className="eyebrow">SYSTEM SUMMARY</span>
          <p>{insights.summary}</p>
          <div className="insight-big-emotion">
            <span>{EMOTION_EMOJI[top.emotion] || "•"}</span>
            <strong>{top.emotion.toUpperCase()}</strong>
            <small>{pct(top.confidence)} calibrated probability</small>
          </div>
        </div>

        <div className="uncertainty-card">
          <div className="uncertainty-head">
            <span className="eyebrow">OPERATIONAL UNCERTAINTY INDEX</span>
            <strong>{(uncertainty.uncertainty_index * 100).toFixed(0)} / 100</strong>
          </div>
          <div className="uncertainty-track"><i style={{ width: `${uncertainty.uncertainty_index * 100}%` }} /></div>
          <div className="uncertainty-metrics">
            <div><span>Entropy</span><strong>{(uncertainty.normalized_entropy * 100).toFixed(0)}%</strong></div>
            <div><span>Top margin</span><strong>{(uncertainty.top_margin * 100).toFixed(1)}%</strong></div>
            <div><span>Disagreement</span><strong>{top.js_divergence.toFixed(4)}</strong></div>
            <div><span>Audio quality</span><strong>{(uncertainty.quality_score * 100).toFixed(0)}%</strong></div>
          </div>
        </div>
      </div>

      <div className="insight-evidence-grid">
        {insights.evidence.map((item, index) => (
          <div className={`insight-evidence-card ${item.strength}`} key={`${item.category}-${item.title}-${index}`}>
            <div className="insight-evidence-top">
              <span>{item.category.replace("_", " ")}</span>
              <b>{item.strength.replace("_", " ")}</b>
            </div>
            <strong>{item.title}</strong>
            <p>{item.detail}</p>
          </div>
        ))}
      </div>

      {insights.personal_baseline && (
        <div className="baseline-comparison-card">
          <div className="result-heading">
            <div>
              <p className="eyebrow">PERSONAL BASELINE · PHASE 8</p>
              <h3>Current voice vs. your baseline</h3>
              <p className="muted-copy">{insights.personal_baseline.summary}</p>
            </div>
            <span className={`result-badge ${insights.personal_baseline.normality === "typical" ? "success" : "warning"}`}>
              {insights.personal_baseline.normality.replaceAll("_", " ").toUpperCase()}
            </span>
          </div>
          <div className="baseline-comparison-metrics">
            <div><span>Deviation index</span><strong>{(insights.personal_baseline.overall_deviation * 100).toFixed(0)}%</strong></div>
            <div><span>Reference samples</span><strong>{insights.personal_baseline.sample_count}</strong></div>
            <div><span>Profile</span><strong>{insights.personal_baseline.profile_id}</strong></div>
          </div>
          <div className="baseline-deviation-list">
            {insights.personal_baseline.metrics.slice(0, 5).map((item) => (
              <div className="baseline-deviation-row" key={item.name}>
                <div>
                  <strong>{item.name.replaceAll("_", " ")}</strong>
                  <small>{item.direction.replaceAll("_", " ")} baseline · {item.significance.replaceAll("_", " ")}</small>
                </div>
                <span>{item.robust_z.toFixed(2)}σ</span>
              </div>
            ))}
          </div>
        </div>
      )}

      <PersonalizedModelPanel profileId={profileId} result={result} sessionId={sessionId} />

      <div className="insight-lower-grid">
        <div className="insight-factor-card">
          <div className="feature-section-heading">
            <span>UNCERTAINTY FACTORS</span>
            <small>interpretable signals</small>
          </div>
          <div className="insight-factor-list">
            {uncertainty.factors.map((factor) => (
              <div className="insight-factor-row" key={factor.name}>
                <div>
                  <strong>{factor.name.replaceAll("_", " ")}</strong>
                  <small>{factor.interpretation}</small>
                </div>
                <span>{factor.value.toFixed(3)}</span>
              </div>
            ))}
          </div>
          <p className="insight-recommendation">{uncertainty.recommendation}</p>
        </div>

        <div className="insight-factor-card">
          <div className="feature-section-heading">
            <span>ALTERNATIVES</span>
            <small>next-highest probabilities</small>
          </div>
          <div className="emotion-probabilities insight-alternatives">
            {insights.alternatives.map((item) => (
              <div className="emotion-row" key={item.emotion}>
                <span>{item.emotion}</span>
                <div className="emotion-track"><i style={{ width: `${Math.max(2, item.probability * 100)}%` }} /></div>
                <strong>{pct(item.probability)}</strong>
              </div>
            ))}
          </div>
          <p className="model-note">{insights.caveat}</p>
        </div>
      </div>
    </div>
  );
}
