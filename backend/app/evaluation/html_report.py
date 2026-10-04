from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_benchmark_html(report: dict[str, Any], destination: str | Path) -> Path:
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    model_rows = []
    for name, metrics in report.get("comparison", {}).items():
        model_rows.append(
            "<tr>"
            f"<td>{html.escape(name)}</td>"
            f"<td>{metrics.get('accuracy', float('nan')):.4f}</td>"
            f"<td>{metrics.get('balanced_accuracy', float('nan')):.4f}</td>"
            f"<td>{metrics.get('macro_f1', float('nan')):.4f}</td>"
            f"<td>{metrics.get('nll', float('nan')):.4f}</td>"
            f"<td>{metrics.get('brier', float('nan')):.4f}</td>"
            f"<td>{metrics.get('ece', float('nan')):.4f}</td>"
            "</tr>"
        )
    payload = json.dumps(report, indent=2)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CEREBRO Evaluation Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{{font-family:Inter,system-ui,sans-serif;max-width:1180px;margin:40px auto;padding:0 24px;line-height:1.5}}
h1{{margin-bottom:4px}} .muted{{color:#666}}
table{{width:100%;border-collapse:collapse;margin:24px 0}} th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}
pre{{background:#f5f5f5;padding:18px;overflow:auto;border-radius:10px}}
.good{{font-weight:600}}
</style>
</head>
<body>
<h1>CEREBRO — Evaluation Report</h1>
<p class="muted">Dataset: {html.escape(str(report.get('dataset','')))} · Test records: {report.get('test_records',0)} · Speaker-independent: {report.get('integrity', {}).get('speaker_leakage') is False}</p>
<h2>Model comparison</h2>
<table>
<thead><tr><th>Model</th><th>Accuracy</th><th>Balanced Acc.</th><th>Macro F1</th><th>NLL</th><th>Brier</th><th>ECE</th></tr></thead>
<tbody>{''.join(model_rows)}</tbody>
</table>
<h2>Integrity</h2>
<pre>{html.escape(json.dumps(report.get('integrity', {}), indent=2))}</pre>
<h2>Speaker-level results</h2>
<pre>{html.escape(json.dumps(report.get('speaker_level', {}), indent=2))}</pre>
<h2>Full machine-readable payload</h2>
<pre>{html.escape(payload)}</pre>
</body>
</html>"""
    destination.write_text(document, encoding="utf-8")
    return destination
