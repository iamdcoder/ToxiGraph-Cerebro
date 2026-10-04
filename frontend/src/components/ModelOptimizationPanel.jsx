import { useEffect, useState } from "react";
import { getOptimizationReport, getOptimizationStatus } from "../api";

function metric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "—";
}

export default function ModelOptimizationPanel() {
  const [status, setStatus] = useState(null);
  const [report, setReport] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const current = await getOptimizationStatus();
        if (cancelled) return;
        setStatus(current);
        if (current.available) {
          const payload = await getOptimizationReport();
          if (!cancelled) setReport(payload);
        }
      } catch (err) {
        if (!cancelled) setError(err.message || "Optimization status unavailable.");
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="evaluation-dashboard panel-card optimization-panel">
      <div className="evaluation-header">
        <div>
          <div className="section-kicker">RESEARCH / MODEL OPTIMIZATION</div>
          <h2>Speaker-independent tuning</h2>
          <p className="muted-copy">
            Hyperparameters are selected with grouped cross-validation on training speakers only.
          </p>
        </div>
        <span className={`build-chip ${status?.available ? "" : "muted"}`}>
          {status?.available ? "REAL OPTIMIZATION" : "OPTIMIZATION PENDING"}
        </span>
      </div>

      {error && <p className="muted-copy">{error}</p>}

      {!status && !error && <p className="muted-copy">Loading optimization status…</p>}

      {status && !status.available && !error && (
        <div className="evaluation-empty">
          <p className="muted-copy">Run the real-data optimization command to populate this panel. No synthetic scores are shown.</p>
        </div>
      )}

      {report && (
        <>
          <div className="metric-grid">
            <div className="metric-item"><span>SELECTED MODEL</span><strong>{report.selected_model?.replaceAll("_", " ") || "—"}</strong></div>
            <div className="metric-item"><span>CV FOLDS</span><strong>{report.cv?.folds ?? "—"}</strong></div>
            <div className="metric-item"><span>TEST MACRO F1</span><strong>{metric(report.test?.macro_f1)}</strong></div>
            <div className="metric-item"><span>TEST ACCURACY</span><strong>{metric(report.test?.accuracy)}</strong></div>
          </div>

          <div className="evaluation-table-wrap">
            <table className="evaluation-table">
              <thead><tr><th>Candidate</th><th>CV F1</th><th>Validation F1</th><th>Validation Acc.</th></tr></thead>
              <tbody>
                {Object.entries(report.candidate_results || {}).map(([name, item]) => (
                  <tr key={name}>
                    <td>{name.replaceAll("_", " ")}</td>
                    <td>{metric(item.cv?.best_macro_f1)}</td>
                    <td>{metric(item.validation?.macro_f1)}</td>
                    <td>{metric(item.validation?.accuracy)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </section>
  );
}
