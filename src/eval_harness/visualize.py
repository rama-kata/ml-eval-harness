"""Generate SVG charts and an interactive HTML dashboard from eval results."""

import json
import math
import sqlite3
import sys
from pathlib import Path

# -- Color palette (dark theme) --
BG = "#0d1117"
BG_CARD = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
ACCENT_COLORS = ["#58a6ff", "#3fb950", "#d29922", "#f85149", "#bc8cff", "#39d2c0", "#f778ba"]


def load_runs(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, model, dataset, accuracy, contains_rate, avg_token_f1, "
        "avg_latency_ms, p95_latency_ms, judge_correct_rate, judge_partial_rate, "
        "raw_results, created_at FROM runs ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


# ── SVG Generation (for README) ──────────────────────────────────────────────


def _svg_bar_chart(runs: list[dict], width: int = 800, height: int = 400) -> str:
    """Horizontal grouped bar chart comparing models across metrics."""
    metrics = [
        ("accuracy", "Exact Match"),
        ("contains_rate", "Contains"),
        ("avg_token_f1", "Token F1"),
    ]
    if any(r.get("judge_correct_rate") is not None for r in runs):
        metrics.append(("judge_correct_rate", "Judge Correct"))

    n_models = len(runs)
    n_metrics = len(metrics)
    margin = {"top": 60, "right": 30, "bottom": 80, "left": 160}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    group_h = chart_h / n_metrics
    bar_h = min(group_h / (n_models + 1) * 0.8, 28)
    group_padding = (group_h - bar_h * n_models) / 2

    bars_svg = []
    labels_svg = []
    gridlines = []

    # Grid lines
    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = margin["left"] + tick * chart_w
        gridlines.append(
            f'<line x1="{x}" y1="{margin["top"]}" x2="{x}" '
            f'y2="{height - margin["bottom"]}" stroke="{BORDER}" stroke-width="1"/>'
        )
        labels_svg.append(
            f'<text x="{x}" y="{height - margin["bottom"] + 20}" '
            f'text-anchor="middle" fill="{TEXT_DIM}" font-size="12">{tick:.0%}</text>'
        )

    for mi, (key, label) in enumerate(metrics):
        group_y = margin["top"] + mi * group_h
        # Metric label
        labels_svg.append(
            f'<text x="{margin["left"] - 10}" y="{group_y + group_h / 2 + 5}" '
            f'text-anchor="end" fill="{TEXT}" font-size="13" font-weight="500">{label}</text>'
        )

        for ri, run in enumerate(runs):
            val = run.get(key) or 0
            color = ACCENT_COLORS[ri % len(ACCENT_COLORS)]
            y = group_y + group_padding + ri * bar_h
            w = val * chart_w

            bars_svg.append(
                f'<rect x="{margin["left"]}" y="{y}" width="{w}" height="{bar_h * 0.85}" '
                f'rx="3" fill="{color}" opacity="0.85">'
                f'<animate attributeName="width" from="0" to="{w}" dur="0.6s" fill="freeze"/>'
                f'</rect>'
            )
            if val > 0.08:
                bars_svg.append(
                    f'<text x="{margin["left"] + w - 8}" y="{y + bar_h * 0.55}" '
                    f'text-anchor="end" fill="{BG}" font-size="11" font-weight="600">'
                    f'{val:.1%}</text>'
                )

    # Legend
    legend_svg = []
    lx = margin["left"]
    for ri, run in enumerate(runs):
        color = ACCENT_COLORS[ri % len(ACCENT_COLORS)]
        name = run["model"]
        legend_svg.append(
            f'<rect x="{lx}" y="{height - 30}" width="12" height="12" rx="2" fill="{color}"/>'
        )
        legend_svg.append(
            f'<text x="{lx + 18}" y="{height - 19}" fill="{TEXT}" font-size="12">{name}</text>'
        )
        lx += len(name) * 8 + 40

    title = (
        f'<text x="{width / 2}" y="30" text-anchor="middle" fill="{TEXT}" '
        f'font-size="16" font-weight="600">Model Comparison — Accuracy Metrics</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" rx="8" fill="{BG}"/>'
        f'{title}'
        f'{"".join(gridlines)}'
        f'{"".join(bars_svg)}'
        f'{"".join(labels_svg)}'
        f'{"".join(legend_svg)}'
        f'</svg>'
    )


def _svg_latency_chart(runs: list[dict], width: int = 800, height: int = 300) -> str:
    """Bar chart for latency comparison."""
    margin = {"top": 60, "right": 30, "bottom": 60, "left": 80}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    max_lat = max(r["avg_latency_ms"] for r in runs) * 1.2
    bar_w = min(chart_w / (len(runs) * 2), 80)
    gap = (chart_w - bar_w * len(runs)) / (len(runs) + 1)

    bars = []
    labels = []

    # Y-axis gridlines
    for i in range(5):
        val = max_lat * i / 4
        y = margin["top"] + chart_h - (val / max_lat * chart_h)
        bars.append(
            f'<line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" '
            f'y2="{y}" stroke="{BORDER}" stroke-width="1"/>'
        )
        labels.append(
            f'<text x="{margin["left"] - 10}" y="{y + 4}" text-anchor="end" '
            f'fill="{TEXT_DIM}" font-size="11">{val:.0f}ms</text>'
        )

    for i, run in enumerate(runs):
        color = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        x = margin["left"] + gap + i * (bar_w + gap)
        bar_h = (run["avg_latency_ms"] / max_lat) * chart_h
        y = margin["top"] + chart_h - bar_h

        p95 = run.get("p95_latency_ms") or run["avg_latency_ms"]
        p95_h = (p95 / max_lat) * chart_h
        p95_y = margin["top"] + chart_h - p95_h

        # P95 whisker
        cx = x + bar_w / 2
        bars.append(
            f'<line x1="{cx}" y1="{p95_y}" x2="{cx}" y2="{y}" '
            f'stroke="{color}" stroke-width="2" opacity="0.5"/>'
        )
        bars.append(
            f'<line x1="{cx - 8}" y1="{p95_y}" x2="{cx + 8}" y2="{p95_y}" '
            f'stroke="{color}" stroke-width="2" opacity="0.5"/>'
        )

        # Main bar
        bars.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'rx="3" fill="{color}" opacity="0.85">'
            f'<animate attributeName="height" from="0" to="{bar_h}" dur="0.6s" fill="freeze"/>'
            f'<animate attributeName="y" from="{margin["top"] + chart_h}" to="{y}" dur="0.6s" fill="freeze"/>'
            f'</rect>'
        )

        # Value label
        bars.append(
            f'<text x="{cx}" y="{y - 8}" text-anchor="middle" fill="{TEXT}" '
            f'font-size="12" font-weight="600">{run["avg_latency_ms"]:.0f}ms</text>'
        )

        # Model name
        labels.append(
            f'<text x="{cx}" y="{margin["top"] + chart_h + 20}" text-anchor="middle" '
            f'fill="{TEXT}" font-size="12">{run["model"]}</text>'
        )

    title = (
        f'<text x="{width / 2}" y="30" text-anchor="middle" fill="{TEXT}" '
        f'font-size="16" font-weight="600">Latency Comparison (avg + p95 whisker)</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" rx="8" fill="{BG}"/>'
        f'{title}'
        f'{"".join(bars)}'
        f'{"".join(labels)}'
        f'</svg>'
    )


def _svg_radar_chart(runs: list[dict], width: int = 500, height: int = 500) -> str:
    """Radar/spider chart comparing models across all metrics."""
    metrics = [
        ("accuracy", "Exact Match"),
        ("contains_rate", "Contains"),
        ("avg_token_f1", "Token F1"),
    ]
    if any(r.get("judge_correct_rate") is not None for r in runs):
        metrics.append(("judge_correct_rate", "Judge"))

    # Normalize latency inversely (lower is better, map to 0-1)
    max_lat = max(r["avg_latency_ms"] for r in runs) * 1.5
    metrics.append(("_speed", "Speed"))

    n = len(metrics)
    cx, cy = width / 2, height / 2 + 10
    radius = min(width, height) / 2 - 70

    angle_step = 2 * math.pi / n
    angles = [-math.pi / 2 + i * angle_step for i in range(n)]

    def polar(angle, r):
        return cx + r * math.cos(angle), cy + r * math.sin(angle)

    svg_parts = []

    # Grid rings
    for ring in [0.25, 0.5, 0.75, 1.0]:
        points = " ".join(f"{polar(a, radius * ring)[0]},{polar(a, radius * ring)[1]}" for a in angles)
        svg_parts.append(
            f'<polygon points="{points}" fill="none" stroke="{BORDER}" stroke-width="1"/>'
        )

    # Axis lines and labels
    for i, (_, label) in enumerate(metrics):
        px, py = polar(angles[i], radius)
        lx, ly = polar(angles[i], radius + 25)
        svg_parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{px}" y2="{py}" stroke="{BORDER}" stroke-width="1"/>'
        )
        anchor = "middle"
        if lx < cx - 10:
            anchor = "end"
        elif lx > cx + 10:
            anchor = "start"
        svg_parts.append(
            f'<text x="{lx}" y="{ly + 4}" text-anchor="{anchor}" fill="{TEXT}" '
            f'font-size="12" font-weight="500">{label}</text>'
        )

    # Data polygons
    for ri, run in enumerate(runs):
        color = ACCENT_COLORS[ri % len(ACCENT_COLORS)]
        values = []
        for key, _ in metrics:
            if key == "_speed":
                values.append(1.0 - (run["avg_latency_ms"] / max_lat))
            else:
                values.append(run.get(key) or 0)

        points = " ".join(
            f"{polar(angles[i], radius * v)[0]},{polar(angles[i], radius * v)[1]}"
            for i, v in enumerate(values)
        )
        svg_parts.append(
            f'<polygon points="{points}" fill="{color}" fill-opacity="0.15" '
            f'stroke="{color}" stroke-width="2"/>'
        )
        # Dots
        for i, v in enumerate(values):
            dx, dy = polar(angles[i], radius * v)
            svg_parts.append(
                f'<circle cx="{dx}" cy="{dy}" r="4" fill="{color}"/>'
            )

    # Legend
    lx = 20
    for ri, run in enumerate(runs):
        color = ACCENT_COLORS[ri % len(ACCENT_COLORS)]
        ly = height - 25
        svg_parts.append(
            f'<rect x="{lx}" y="{ly}" width="12" height="12" rx="2" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{lx + 18}" y="{ly + 11}" fill="{TEXT}" font-size="12">{run["model"]}</text>'
        )
        lx += len(run["model"]) * 8 + 40

    title = (
        f'<text x="{width / 2}" y="28" text-anchor="middle" fill="{TEXT}" '
        f'font-size="16" font-weight="600">Model Profile</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" rx="8" fill="{BG}"/>'
        f'{title}'
        f'{"".join(svg_parts)}'
        f'</svg>'
    )


# ── HTML Dashboard ────────────────────────────────────────────────────────────


def _build_html(runs: list[dict]) -> str:
    """Build a self-contained interactive HTML dashboard."""
    # Prepare per-item details
    details_json = {}
    for run in runs:
        raw = json.loads(run["raw_results"]) if run.get("raw_results") else {}
        details_json[run["id"]] = {
            "model": run["model"],
            "dataset": run["dataset"],
            "items": raw.get("details", []),
        }

    radar_svg = _svg_radar_chart(runs)
    bar_svg = _svg_bar_chart(runs)
    latency_svg = _svg_latency_chart(runs)

    # Build the results table rows
    table_rows = ""
    for run in runs:
        judge_col = ""
        if run.get("judge_correct_rate") is not None:
            judge_col = f'<td>{run["judge_correct_rate"]:.1%}</td>'
        else:
            judge_col = '<td class="dim">—</td>'

        table_rows += f"""
        <tr data-run-id="{run['id']}" onclick="showDetails({run['id']})">
            <td class="model-name">{run['model']}</td>
            <td>{run['accuracy']:.1%}</td>
            <td>{run['contains_rate']:.1%}</td>
            <td>{run.get('avg_token_f1', 0):.3f}</td>
            <td>{run['avg_latency_ms']:.0f}ms</td>
            {judge_col}
            <td class="dim">{run['created_at'][:19]}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ml-eval-harness — Results Dashboard</title>
<style>
  :root {{
    --bg: {BG}; --card: {BG_CARD}; --border: {BORDER};
    --text: {TEXT}; --dim: {TEXT_DIM};
    --blue: #58a6ff; --green: #3fb950; --yellow: #d29922;
    --red: #f85149; --purple: #bc8cff;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 14px; line-height: 1.5;
    padding: 2rem; max-width: 1200px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.8rem; font-weight: 600; margin-bottom: 0.5rem; }}
  h1 span {{ color: var(--blue); }}
  .subtitle {{ color: var(--dim); margin-bottom: 2rem; }}
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .chart-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.5rem; overflow: hidden;
  }}
  .chart-card.full {{ grid-column: 1 / -1; }}
  .chart-card svg {{ width: 100%; height: auto; }}
  table {{
    width: 100%; border-collapse: collapse; background: var(--card);
    border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
  }}
  th {{
    text-align: left; padding: 12px 16px; font-weight: 600; font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.05em; color: var(--dim);
    border-bottom: 1px solid var(--border); background: var(--card);
  }}
  td {{ padding: 10px 16px; border-bottom: 1px solid var(--border); }}
  tr {{ cursor: pointer; transition: background 0.15s; }}
  tr:hover {{ background: rgba(88,166,255,0.06); }}
  tr.active {{ background: rgba(88,166,255,0.1); }}
  .model-name {{ font-weight: 600; color: var(--blue); }}
  .dim {{ color: var(--dim); }}
  .section-title {{
    font-size: 1.1rem; font-weight: 600; margin: 2rem 0 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
  }}
  #details-panel {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.5rem; margin-top: 1rem; display: none;
  }}
  #details-panel.visible {{ display: block; }}
  .detail-item {{
    padding: 1rem; margin-bottom: 0.75rem; border-radius: 8px;
    border: 1px solid var(--border); background: var(--bg);
  }}
  .detail-item .prompt {{ color: var(--blue); font-weight: 500; margin-bottom: 0.5rem; }}
  .detail-item .response {{ margin: 0.5rem 0; padding: 0.5rem; background: var(--card); border-radius: 4px; }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 600; margin-right: 4px;
  }}
  .badge.correct {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .badge.partial {{ background: rgba(210,153,34,0.15); color: var(--yellow); }}
  .badge.wrong {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .badge.metric {{ background: rgba(88,166,255,0.1); color: var(--blue); }}
  .f1-bar {{
    height: 4px; border-radius: 2px; background: var(--border);
    margin-top: 4px; overflow: hidden;
  }}
  .f1-bar-fill {{ height: 100%; border-radius: 2px; background: var(--green); transition: width 0.3s; }}
  @media (max-width: 768px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1><span>ml-eval-harness</span> — Results</h1>
<p class="subtitle">Comparing local LLMs via Ollama · {len(runs)} run(s) recorded</p>

<div class="charts">
  <div class="chart-card">{radar_svg}</div>
  <div class="chart-card">{latency_svg}</div>
  <div class="chart-card full">{bar_svg}</div>
</div>

<div class="section-title">All Runs</div>
<table>
  <thead>
    <tr>
      <th>Model</th><th>Exact Match</th><th>Contains</th>
      <th>Token F1</th><th>Avg Latency</th><th>Judge</th><th>Date</th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>

<div class="section-title">Per-Item Details <span class="dim" style="font-weight:400;font-size:0.85rem">(click a run above)</span></div>
<div id="details-panel"></div>

<script>
const DATA = {json.dumps(details_json)};

function showDetails(runId) {{
  const panel = document.getElementById('details-panel');
  const run = DATA[runId];
  if (!run || !run.items.length) {{
    panel.innerHTML = '<p class="dim">No per-item details for this run.</p>';
    panel.classList.add('visible');
    return;
  }}

  document.querySelectorAll('tr.active').forEach(r => r.classList.remove('active'));
  document.querySelector(`tr[data-run-id="${{runId}}"]`)?.classList.add('active');

  let html = `<h3 style="margin-bottom:1rem">${{run.model}} — ${{run.items.length}} items</h3>`;
  run.items.forEach((item, i) => {{
    const judgeBadge = item.judge_verdict
      ? `<span class="badge ${{item.judge_verdict}}">${{item.judge_verdict}}</span>
         <span class="dim" style="font-size:12px">${{item.judge_reason || ''}}</span>`
      : '';

    const f1Pct = ((item.token_f1 || 0) * 100).toFixed(0);

    html += `
      <div class="detail-item">
        <div class="prompt">${{i+1}}. ${{item.prompt}}</div>
        <div><strong>Expected:</strong> ${{item.expected}}</div>
        <div class="response"><strong>Response:</strong> ${{item.response}}</div>
        <div>
          <span class="badge metric">exact: ${{item.exact_match ? '✓' : '✗'}}</span>
          <span class="badge metric">contains: ${{item.contains ? '✓' : '✗'}}</span>
          <span class="badge metric">F1: ${{f1Pct}}%</span>
          <span class="badge metric">${{(item.latency_s * 1000).toFixed(0)}}ms</span>
          ${{judgeBadge}}
        </div>
        <div class="f1-bar"><div class="f1-bar-fill" style="width:${{f1Pct}}%"></div></div>
      </div>`;
  }});

  panel.innerHTML = html;
  panel.classList.add('visible');
}}
</script>
</body>
</html>"""


# ── Main entry point ─────────────────────────────────────────────────────────


def generate(db_path: str = "results.db", output_dir: str = "docs"):
    """Generate all visualizations."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(db_path)
    if not runs:
        print("No runs found in database.")
        return

    print(f"Generating visualizations for {len(runs)} run(s)...")

    # SVGs for README
    radar = _svg_radar_chart(runs)
    (out / "radar.svg").write_text(radar)
    print(f"  → {out / 'radar.svg'}")

    bar = _svg_bar_chart(runs)
    (out / "comparison.svg").write_text(bar)
    print(f"  → {out / 'comparison.svg'}")

    latency = _svg_latency_chart(runs)
    (out / "latency.svg").write_text(latency)
    print(f"  → {out / 'latency.svg'}")

    # HTML dashboard
    html = _build_html(runs)
    (out / "index.html").write_text(html)
    print(f"  → {out / 'index.html'}")

    print("Done.")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "results.db"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs"
    generate(db, out)
