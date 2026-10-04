import { useEffect, useState } from "react";
import { getRobustnessReport, getRobustnessStatus } from "../api";

function formatMetric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
}

function deltaClass(value) {
  if (typeof value !== "number") return "";
  return value < -0.05 ? "robustness-negative" : value > 0.01 ? "robustness-positive" : "";
}

export default function RobustnessDashboard() {
  const [status, setStatus] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const current = await getRobustnessStatus();
        if (cancelled) return;
        setStatus(current);
        if (current.available) {
          const payload = await getRobustnessReport();
          if (!cancelled) setReport(payload);
        }
      } catch (err) {
        if (!cancelled) setError(err?.message || "Robustness status unavailable.");
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <section className="robustness-dashboard panel-card">
        <div className="section-kicker">RESEARCH / ROBUSTNESS LAB</div>
        <h2>Robustness analysis is unavailable.</h2>
        <p className="muted-copy">{error}</p>
      </section>
    );
  }

  if (!status) {
    return (
      <section className="robustness-dashboard panel-card">
        <div className="section-kicker">RESEARCH / ROBUSTNESS LAB</div>
        <h2>Loading robustness status…</h2>
      </section>
    );
  }

  if (!status.available || !report) {
    return (
      <section className="robustness-dashboard panel-card">
        <div className="robustness-header">
          <div>
            <div className="section-kicker">RESEARCH / ROBUSTNESS LAB</div>
            <h2>How resilient is CEREBRO?</h2>
            <p className="muted-copy">
              This panel reports measured performance under controlled audio stress tests only after a real held-out benchmark has been generated.
            </p>
          </div>
          <span className="build-chip muted">STRESS TEST PENDING</span>
        </div>
        <div className="robustness-condition-chips">
          <span>NOISE</span><span>GAIN</span><span>CLIPPING</span><span>REVERB</span><span>SPEED</span><span>BANDWIDTH</span>
        </div>
      </section>
    );
  }

  return (
    <section className="robustness-dashboard panel-card">
      <div className="robustness-header">
        <div>
          <div className="section-kicker">RESEARCH / ROBUSTNESS LAB</div>
          <h2>{report.dataset}</h2>
          <p className="muted-copy">
            {report.records} held-out recordings · {report.conditions?.length || 0} controlled stress conditions
          </p>
        </div>
        <span className="build-chip">REAL STRESS TEST</span>
      </div>

      <div className="robustness-summary">
        <div><span>Clean macro F1</span><strong>{formatMetric(report.clean_metrics?.macro_f1)}</strong></div>
        <div><span>Clean accuracy</span><strong>{formatMetric(report.clean_metrics?.accuracy)}</strong></div>
        <div><span>Unique speakers</span><strong>{report.integrity?.unique_speakers ?? "—"}</strong></div>
      </div>

      <div className="evaluation-table-wrap">
        <table className="evaluation-table robustness-table">
          <thead><tr><th>Condition</th><th>Macro F1</th><th>Δ vs clean</th><th>Confidence</th><th>Clean agreement</th><th>Quality</th></tr></thead>
          <tbody>
            {report.conditions?.map((item) => (
              <tr key={item.condition}>
                <td title={item.description}>{item.condition.replaceAll("_", " ")}</td>
                <td>{formatMetric(item.macro_f1)}</td>
                <td className={deltaClass(item.delta_macro_f1_vs_clean)}>{formatMetric(item.delta_macro_f1_vs_clean)}</td>
                <td>{formatMetric(item.mean_confidence)}</td>
                <td>{formatMetric(item.clean_agreement)}</td>
                <td>{formatMetric(item.quality_score_mean)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="robustness-note">
        These perturbations approximate common recording failures. They are controlled stress tests, not substitutes for validation on real noisy environments.
      </p>
    </section>
  );
}
