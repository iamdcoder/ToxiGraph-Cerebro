from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_robustness_html(payload: dict[str, Any], output_path: str | Path) -> Path:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    conditions = payload.get("conditions", [])
    clean = payload.get("clean_metrics", {})
    rows = []
    for item in conditions:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('condition', '')))}</td>"
            f"<td>{html.escape(str(item.get('family', '')))}</td>"
            f"<td>{float(item.get('macro_f1', 0.0)):.4f}</td>"
            f"<td>{float(item.get('delta_macro_f1_vs_clean', 0.0)):+.4f}</td>"
            f"<td>{float(item.get('mean_confidence', 0.0)):.4f}</td>"
            f"<td>{float(item.get('clean_agreement', 0.0)):.4f}</td>"
            f"<td>{float(item.get('quality_score_mean', 0.0)):.4f}</td>"
            "</tr>"
        )
    document = f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
<title>CEREBRO Robustness Report</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;background:#080b10;color:#e9edf5;margin:0;padding:32px;line-height:1.5}}
main{{max-width:1200px;margin:auto}}
.card{{border:1px solid #202633;border-radius:18px;background:#0e131b;padding:22px;margin:0 0 18px}}
h1{{margin:0 0 8px;font-size:30px}}h2{{font-size:18px}}
.muted{{color:#8d97a9}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px;border-bottom:1px solid #202633;text-align:right}}th:first-child,td:first-child{{text-align:left}}
th{{color:#8d97a9;text-transform:uppercase;letter-spacing:.08em;font-size:10px}}
.good{{color:#b7f3df}}.warn{{color:#ffd390}}
pre{{white-space:pre-wrap;overflow:auto;color:#adb6c5}}
</style>
</head>
<body>
<main>
<div class=\"card\">
<h1>CEREBRO Robustness Report</h1>
<p class=\"muted\">Dataset: {html.escape(str(payload.get('dataset', '')))} · Records: {int(payload.get('records', 0))} · Seed: {int(payload.get('seed', 42))}</p>
<p>Clean macro-F1: <strong>{float(clean.get('macro_f1', 0.0)):.4f}</strong></p>
</div>
<div class=\"card\">
<h2>Controlled perturbation results</h2>
<table>
<thead><tr><th>Condition</th><th>Family</th><th>Macro F1</th><th>Δ vs clean</th><th>Mean confidence</th><th>Clean agreement</th><th>Quality score</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</div>
<div class=\"card\">
<h2>Integrity</h2>
<pre>{html.escape(json.dumps(payload.get('integrity', {}), indent=2))}</pre>
</div>
</main>
</body></html>"""
    destination.write_text(document, encoding="utf-8")
    return destination
