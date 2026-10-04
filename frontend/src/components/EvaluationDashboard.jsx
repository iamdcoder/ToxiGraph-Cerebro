import { useEffect, useState } from "react";
import { getEvaluationReport, getEvaluationStatus } from "../api";

const metricLabels = [
  ["accuracy", "Accuracy"],
  ["balanced_accuracy", "Balanced accuracy"],
  ["macro_f1", "Macro F1"],
  ["nll", "NLL"],
  ["brier", "Brier"],
  ["ece", "ECE"],
];

function formatMetric(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value.toFixed(4);
}

export default function EvaluationDashboard() {
  const [status, setStatus] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const current = await getEvaluationStatus();
        if (cancelled) return;
        setStatus(current);
        if (current.available) {
          const payload = await getEvaluationReport();
          if (!cancelled) setReport(payload);
        }
      } catch (err) {
        if (!cancelled) setError(err.message || "Evaluation status unavailable.");
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) {
    return (
      <section className="evaluation-dashboard panel-card">
        <div className="section-kicker">RESEARCH / MODEL PERFORMANCE</div>
        <h2>Evaluation is unavailable.</h2>
        <p className="muted-copy">{error}</p>
      </section>
    );
  }

  if (!status) {
    return (
      <section className="evaluation-dashboard panel-card">
        <div className="section-kicker">RESEARCH / MODEL PERFORMANCE</div>
        <h2>Loading evaluation status…</h2>
      </section>
    );
  }

  if (!status.available || !report) {
    return (
      <section className="evaluation-dashboard panel-card">
        <div className="section-kicker">RESEARCH / MODEL PERFORMANCE</div>
        <div className="evaluation-empty">
          <div>
            <h2>No performance numbers yet.</h2>
            <p className="muted-copy">
              CEREBRO will show real held-out metrics here only after the speaker-independent benchmark has been run.
              No synthetic or placeholder scores are displayed.
            </p>
          </div>
          <span className="build-chip muted">BENCHMARK PENDING</span>
        </div>
      </section>
    );
  }

  return (
    <section className="evaluation-dashboard panel-card">
      <div className="evaluation-header">
        <div>
          <div className="section-kicker">RESEARCH / MODEL PERFORMANCE</div>
          <h2>{report.dataset}</h2>
          <p className="muted-copy">
            {report.test_records} held-out recordings · {report.test_speakers?.length || 0} unseen speakers
          </p>
        </div>
        <span className="build-chip">REAL BENCHMARK</span>
      </div>

      <div className="evaluation-table-wrap">
        <table className="evaluation-table">
          <thead>
            <tr>
              <th>Model</th>
              {metricLabels.map(([, label]) => <th key={label}>{label}</th>)}
            </tr>
          </thead>
          <tbody>
            {Object.entries(report.comparison || {}).map(([name, metrics]) => (
              <tr key={name}>
                <td>{name.replaceAll("_", " ")}</td>
                {metricLabels.map(([key]) => <td key={key}>{formatMetric(metrics[key])}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="evaluation-integrity">
        <strong>Evaluation integrity:</strong>
        <span>speaker leakage = {report.integrity?.speaker_leakage ? "detected" : "none detected"}</span>
      </div>
    </section>
  );
}
