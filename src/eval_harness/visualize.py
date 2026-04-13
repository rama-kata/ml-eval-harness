"""Generate SVG charts and an interactive HTML dashboard from eval results."""

import json
import math
import sqlite3
import sys
from pathlib import Path

# -- Color palette (dark theme, WCAG AA compliant on #0d1117) --
BG = "#0d1117"
BG_CARD = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
ACCENT_COLORS = ["#58a6ff", "#3fb950", "#e3b341", "#f85149", "#bc8cff", "#39d2c0", "#f778ba", "#79c0ff"]


def load_runs(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, model, dataset, accuracy, contains_rate, avg_token_f1, "
        "avg_latency_ms, p95_latency_ms, judge_correct_rate, judge_partial_rate, "
        "raw_results, created_at FROM runs ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def _tooltip(text: str) -> str:
    """Wrap text in an SVG <title> element for native hover tooltip."""
    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<title>{escaped}</title>"


# ── SVG Generation (for README) ──────────────────────────────────────────────


def _svg_bar_chart(runs: list[dict], width: int = 800, height: int = 500) -> str:
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
                f'<g class="bar-group" style="cursor:pointer">'
                f'{_tooltip(f"{run["model"]}: {val:.1%}")}'
                f'<rect x="{margin["left"]}" y="{y}" width="{w}" height="{bar_h * 0.85}" '
                f'rx="3" fill="{color}" opacity="0.85">'
                f'<animate attributeName="width" from="0" to="{w}" dur="0.6s" fill="freeze"/>'
                f'</rect>'
                f'</g>'
            )
            if val > 0.08:
                bars_svg.append(
                    f'<text x="{margin["left"] + w - 8}" y="{y + bar_h * 0.55}" '
                    f'text-anchor="end" fill="{BG}" font-size="11" font-weight="600">'
                    f'{val:.1%}</text>'
                )

    # Legend — wrap to multiple rows if needed
    legend_svg = []
    lx = margin["left"]
    ly = height - 30
    for ri, run in enumerate(runs):
        color = ACCENT_COLORS[ri % len(ACCENT_COLORS)]
        name = run["model"]
        text_w = len(name) * 7.5 + 30
        if lx + text_w > width - margin["right"]:
            lx = margin["left"]
            ly += 18
        legend_svg.append(
            f'<rect x="{lx}" y="{ly}" width="12" height="12" rx="2" fill="{color}"/>'
        )
        legend_svg.append(
            f'<text x="{lx + 18}" y="{ly + 11}" fill="{TEXT}" font-size="11">{name}</text>'
        )
        lx += text_w

    title = (
        f'<text x="{width / 2}" y="30" text-anchor="middle" fill="{TEXT}" '
        f'font-size="16" font-weight="600">Model Comparison — Accuracy Metrics</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height + 20}" '
        f'viewBox="0 0 {width} {height + 20}">'
        f'<rect width="{width}" height="{height + 20}" rx="8" fill="{BG}"/>'
        f'{title}'
        f'{"".join(gridlines)}'
        f'{"".join(bars_svg)}'
        f'{"".join(labels_svg)}'
        f'{"".join(legend_svg)}'
        f'</svg>'
    )


def _svg_latency_chart(runs: list[dict], width: int = 800, height: int = 350) -> str:
    """Bar chart for latency comparison with correct p95 whisker scaling."""
    margin = {"top": 60, "right": 30, "bottom": 80, "left": 80}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    # BUG FIX: scale to the max of p95 OR avg, whichever is higher
    max_lat = max(
        max(r.get("p95_latency_ms") or r["avg_latency_ms"], r["avg_latency_ms"])
        for r in runs
    ) * 1.15

    bar_w = min(chart_w / (len(runs) * 1.8), 65)
    gap = (chart_w - bar_w * len(runs)) / (len(runs) + 1)

    bars = []
    labels = []

    # Y-axis gridlines
    n_ticks = 5
    for i in range(n_ticks + 1):
        val = max_lat * i / n_ticks
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

        cx = x + bar_w / 2

        # P95 whisker
        bars.append(
            f'<g style="cursor:pointer">'
            f'{_tooltip(f"{run["model"]}: avg={run["avg_latency_ms"]:.0f}ms p95={p95:.0f}ms")}'
            f'<line x1="{cx}" y1="{p95_y}" x2="{cx}" y2="{y}" '
            f'stroke="{color}" stroke-width="2" opacity="0.5"/>'
            f'<line x1="{cx - 8}" y1="{p95_y}" x2="{cx + 8}" y2="{p95_y}" '
            f'stroke="{color}" stroke-width="2" opacity="0.5"/>'
            f'</g>'
        )

        # Main bar
        bars.append(
            f'<g style="cursor:pointer">'
            f'{_tooltip(f"{run["model"]}: {run["avg_latency_ms"]:.0f}ms avg")}'
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'rx="3" fill="{color}" opacity="0.85">'
            f'<animate attributeName="height" from="0" to="{bar_h}" dur="0.6s" fill="freeze"/>'
            f'<animate attributeName="y" from="{margin["top"] + chart_h}" to="{y}" dur="0.6s" fill="freeze"/>'
            f'</rect>'
            f'</g>'
        )

        # Value label
        bars.append(
            f'<text x="{cx}" y="{y - 8}" text-anchor="middle" fill="{TEXT}" '
            f'font-size="11" font-weight="600">{run["avg_latency_ms"]:.0f}ms</text>'
        )

        # Model name — rotated for readability with many models
        labels.append(
            f'<text x="{cx}" y="{margin["top"] + chart_h + 14}" text-anchor="end" '
            f'fill="{TEXT}" font-size="11" '
            f'transform="rotate(-35 {cx} {margin["top"] + chart_h + 14})">'
            f'{run["model"]}</text>'
        )

    title = (
        f'<text x="{width / 2}" y="30" text-anchor="middle" fill="{TEXT}" '
        f'font-size="16" font-weight="600">Latency Comparison (avg + p95 whisker)</text>'
    )
    subtitle = (
        f'<text x="{width / 2}" y="48" text-anchor="middle" fill="{TEXT_DIM}" '
        f'font-size="12">Lower is better</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'<rect width="{width}" height="{height}" rx="8" fill="{BG}"/>'
        f'{title}{subtitle}'
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

    # Grid rings with labels
    for ring in [0.25, 0.5, 0.75, 1.0]:
        points = " ".join(f"{polar(a, radius * ring)[0]},{polar(a, radius * ring)[1]}" for a in angles)
        svg_parts.append(
            f'<polygon points="{points}" fill="none" stroke="{BORDER}" stroke-width="1"/>'
        )
        rx, ry = polar(angles[0], radius * ring)
        svg_parts.append(
            f'<text x="{rx + 4}" y="{ry - 4}" fill="{TEXT_DIM}" font-size="9">{ring:.0%}</text>'
        )

    # Axis lines and labels
    for i, (_, label) in enumerate(metrics):
        px, py = polar(angles[i], radius)
        lx, ly = polar(angles[i], radius + 30)
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

    # Data polygons with tooltips
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
        tooltip_text = f'{run["model"]}: exact={run["accuracy"]:.0%} f1={run.get("avg_token_f1", 0):.2f} lat={run["avg_latency_ms"]:.0f}ms'
        svg_parts.append(
            f'<g style="cursor:pointer">'
            f'{_tooltip(tooltip_text)}'
            f'<polygon points="{points}" fill="{color}" fill-opacity="0.12" '
            f'stroke="{color}" stroke-width="2"/>'
            f'</g>'
        )
        for i, v in enumerate(values):
            dx, dy = polar(angles[i], radius * v)
            svg_parts.append(
                f'<circle cx="{dx}" cy="{dy}" r="3.5" fill="{color}" stroke="{BG}" stroke-width="1"/>'
            )

    # Legend — wrap rows
    lx = 20
    ly = height - 25
    for ri, run in enumerate(runs):
        color = ACCENT_COLORS[ri % len(ACCENT_COLORS)]
        name = run["model"]
        text_w = len(name) * 7 + 28
        if lx + text_w > width - 20:
            lx = 20
            ly += 16
        svg_parts.append(
            f'<rect x="{lx}" y="{ly}" width="10" height="10" rx="2" fill="{color}"/>'
        )
        svg_parts.append(
            f'<text x="{lx + 15}" y="{ly + 10}" fill="{TEXT}" font-size="10">{name}</text>'
        )
        lx += text_w

    extra_h = max(0, ly - (height - 25))  # expand if legend wrapped
    total_h = height + extra_h + 10

    title = (
        f'<text x="{width / 2}" y="28" text-anchor="middle" fill="{TEXT}" '
        f'font-size="16" font-weight="600">Model Profile</text>'
    )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{total_h}" '
        f'viewBox="0 0 {width} {total_h}">'
        f'<rect width="{width}" height="{total_h}" rx="8" fill="{BG}"/>'
        f'{title}'
        f'{"".join(svg_parts)}'
        f'</svg>'
    )


# ── HTML Dashboard ────────────────────────────────────────────────────────────


def _build_html(runs: list[dict]) -> str:
    """Build a self-contained interactive HTML dashboard."""
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

    # Summary stats for cards
    best_accuracy = max(runs, key=lambda r: r.get("accuracy", 0))
    fastest = min(runs, key=lambda r: r["avg_latency_ms"])
    best_f1 = max(runs, key=lambda r: r.get("avg_token_f1", 0))

    # Build the results table rows
    table_rows = ""
    for run in runs:
        judge_col = ""
        if run.get("judge_correct_rate") is not None:
            judge_col = f'<td>{run["judge_correct_rate"]:.1%}</td>'
        else:
            judge_col = '<td class="dim">\u2014</td>'

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

    # Model checkboxes for radar filter
    model_checks = ""
    for ri, run in enumerate(runs):
        color = ACCENT_COLORS[ri % len(ACCENT_COLORS)]
        checked = "checked" if ri < 4 else ""
        model_checks += (
            f'<label class="model-check" style="--accent:{color}">'
            f'<input type="checkbox" value="{run["id"]}" {checked} onchange="updateRadar()">'
            f'<span class="check-dot" style="background:{color}"></span>'
            f'{run["model"]}</label>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ml-eval-harness \u2014 Results Dashboard</title>
<style>
  :root {{
    --bg: {BG}; --card: {BG_CARD}; --border: {BORDER};
    --text: {TEXT}; --dim: {TEXT_DIM};
    --blue: #58a6ff; --green: #3fb950; --yellow: #e3b341;
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
  .subtitle {{ color: var(--dim); margin-bottom: 1.5rem; }}

  /* ── Summary Cards ── */
  .summary-cards {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem;
  }}
  .stat-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 10px;
    padding: 1.2rem; text-align: center;
  }}
  .stat-card .stat-value {{
    font-size: 1.6rem; font-weight: 700; margin-bottom: 0.25rem;
  }}
  .stat-card .stat-label {{ color: var(--dim); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }}
  .stat-card .stat-detail {{ color: var(--dim); font-size: 12px; margin-top: 0.3rem; }}
  .stat-card:nth-child(1) .stat-value {{ color: var(--blue); }}
  .stat-card:nth-child(2) .stat-value {{ color: var(--green); }}
  .stat-card:nth-child(3) .stat-value {{ color: var(--purple); }}
  .stat-card:nth-child(4) .stat-value {{ color: var(--yellow); }}

  /* ── Charts ── */
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .chart-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.5rem; overflow: hidden;
  }}
  .chart-card.full {{ grid-column: 1 / -1; }}
  .chart-card svg {{ width: 100%; height: auto; }}

  /* ── Radar filter ── */
  .radar-controls {{ margin-bottom: 0.75rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }}
  .model-check {{
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; cursor: pointer; padding: 3px 8px;
    border-radius: 6px; border: 1px solid var(--border);
    transition: background 0.15s;
  }}
  .model-check:hover {{ background: rgba(255,255,255,0.04); }}
  .model-check input {{ display: none; }}
  .check-dot {{
    width: 8px; height: 8px; border-radius: 50%; opacity: 0.35;
    transition: opacity 0.15s;
  }}
  .model-check input:checked ~ .check-dot {{ opacity: 1; }}

  /* ── Table ── */
  table {{
    width: 100%; border-collapse: collapse; background: var(--card);
    border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
  }}
  th {{
    text-align: left; padding: 12px 16px; font-weight: 600; font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.05em; color: var(--dim);
    border-bottom: 1px solid var(--border); background: var(--card);
    cursor: pointer; user-select: none; white-space: nowrap;
    transition: color 0.15s;
  }}
  th:hover {{ color: var(--text); }}
  th .sort-arrow {{ font-size: 10px; margin-left: 4px; opacity: 0.4; }}
  th.sorted .sort-arrow {{ opacity: 1; color: var(--blue); }}
  td {{ padding: 10px 16px; border-bottom: 1px solid var(--border); }}
  tbody tr {{ cursor: pointer; transition: background 0.15s; }}
  tbody tr:hover {{ background: rgba(88,166,255,0.06); }}
  tbody tr.active {{ background: rgba(88,166,255,0.1); }}
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
  .badge.partial {{ background: rgba(227,179,65,0.15); color: var(--yellow); }}
  .badge.wrong {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .badge.metric {{ background: rgba(88,166,255,0.1); color: var(--blue); }}
  .f1-bar {{
    height: 4px; border-radius: 2px; background: var(--border);
    margin-top: 4px; overflow: hidden;
  }}
  .f1-bar-fill {{ height: 100%; border-radius: 2px; background: var(--green); transition: width 0.3s; }}
  @media (max-width: 768px) {{
    .charts {{ grid-template-columns: 1fr; }}
    .summary-cards {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>

<h1><span>ml-eval-harness</span> \u2014 Results</h1>
<p class="subtitle">Comparing local LLMs via Ollama \u00b7 {len(runs)} run(s) recorded</p>

<!-- Summary Cards -->
<div class="summary-cards">
  <div class="stat-card">
    <div class="stat-value">{len(runs)}</div>
    <div class="stat-label">Total Runs</div>
    <div class="stat-detail">{len(set(r['model'] for r in runs))} unique models</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{best_accuracy['accuracy']:.0%}</div>
    <div class="stat-label">Best Accuracy</div>
    <div class="stat-detail">{best_accuracy['model']}</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{best_f1.get('avg_token_f1', 0):.2f}</div>
    <div class="stat-label">Best Token F1</div>
    <div class="stat-detail">{best_f1['model']}</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{fastest['avg_latency_ms']:.0f}ms</div>
    <div class="stat-label">Fastest</div>
    <div class="stat-detail">{fastest['model']}</div>
  </div>
</div>

<div class="charts">
  <div class="chart-card">
    <div class="radar-controls">{model_checks}</div>
    <div id="radar-container">{radar_svg}</div>
  </div>
  <div class="chart-card">{latency_svg}</div>
  <div class="chart-card full">{bar_svg}</div>
</div>

<div class="section-title">All Runs</div>
<table id="results-table">
  <thead>
    <tr>
      <th data-col="0" data-type="str" onclick="sortTable(this)">Model <span class="sort-arrow">\u25B2</span></th>
      <th data-col="1" data-type="num" onclick="sortTable(this)">Exact Match <span class="sort-arrow">\u25B2</span></th>
      <th data-col="2" data-type="num" onclick="sortTable(this)">Contains <span class="sort-arrow">\u25B2</span></th>
      <th data-col="3" data-type="num" onclick="sortTable(this)">Token F1 <span class="sort-arrow">\u25B2</span></th>
      <th data-col="4" data-type="num" onclick="sortTable(this)">Avg Latency <span class="sort-arrow">\u25B2</span></th>
      <th data-col="5" data-type="num" onclick="sortTable(this)">Judge <span class="sort-arrow">\u25B2</span></th>
      <th data-col="6" data-type="str" onclick="sortTable(this)">Date <span class="sort-arrow">\u25B2</span></th>
    </tr>
  </thead>
  <tbody>{table_rows}</tbody>
</table>

<div class="section-title">Per-Item Details <span class="dim" style="font-weight:400;font-size:0.85rem">(click a run above)</span></div>
<div id="details-panel"></div>

<script>
const DATA = {json.dumps(details_json)};
const ALL_RUNS = {json.dumps([{
    "id": r["id"], "model": r["model"], "accuracy": r.get("accuracy", 0),
    "contains_rate": r.get("contains_rate", 0), "avg_token_f1": r.get("avg_token_f1", 0),
    "avg_latency_ms": r["avg_latency_ms"],
    "judge_correct_rate": r.get("judge_correct_rate")
} for r in runs])};
const COLORS = {json.dumps(ACCENT_COLORS)};

// ── Sortable Table ──
let sortState = {{ col: null, asc: true }};
function sortTable(th) {{
  const col = parseInt(th.dataset.col);
  const type = th.dataset.type;
  const tbody = document.querySelector('#results-table tbody');
  const rows = Array.from(tbody.querySelectorAll('tr'));

  if (sortState.col === col) {{ sortState.asc = !sortState.asc; }}
  else {{ sortState.col = col; sortState.asc = type === 'str'; }}

  document.querySelectorAll('#results-table th').forEach(h => h.classList.remove('sorted'));
  th.classList.add('sorted');
  th.querySelector('.sort-arrow').textContent = sortState.asc ? '\u25B2' : '\u25BC';

  rows.sort((a, b) => {{
    let av = a.cells[col].textContent.trim();
    let bv = b.cells[col].textContent.trim();
    if (type === 'num') {{
      av = parseFloat(av) || 0;
      bv = parseFloat(bv) || 0;
    }}
    let cmp = av < bv ? -1 : av > bv ? 1 : 0;
    return sortState.asc ? cmp : -cmp;
  }});

  rows.forEach(r => tbody.appendChild(r));
}}

// ── Radar Filter ──
function updateRadar() {{
  const checked = Array.from(document.querySelectorAll('.model-check input:checked'))
    .map(cb => parseInt(cb.value));
  const filtered = ALL_RUNS.filter(r => checked.includes(r.id));
  if (filtered.length === 0) return;

  // Rebuild radar SVG client-side
  const container = document.getElementById('radar-container');
  const W = 500, H = 500;
  const cx = W/2, cy = H/2 + 10, R = W/2 - 70;

  const metrics = ['accuracy', 'contains_rate', 'avg_token_f1'];
  const labels = ['Exact Match', 'Contains', 'Token F1'];
  if (filtered.some(r => r.judge_correct_rate != null)) {{
    metrics.push('judge_correct_rate'); labels.push('Judge');
  }}
  const maxLat = Math.max(...filtered.map(r => r.avg_latency_ms)) * 1.5;
  metrics.push('_speed'); labels.push('Speed');

  const n = metrics.length;
  const angles = Array.from({{length: n}}, (_, i) => -Math.PI/2 + i * 2 * Math.PI / n);
  const polar = (a, r) => [cx + r * Math.cos(a), cy + r * Math.sin(a)];

  let svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${{W}}" height="${{H}}" viewBox="0 0 ${{W}} ${{H}}">`;
  svg += `<rect width="${{W}}" height="${{H}}" rx="8" fill="{BG}"/>`;
  svg += `<text x="${{W/2}}" y="28" text-anchor="middle" fill="{TEXT}" font-size="16" font-weight="600">Model Profile</text>`;

  // Grid
  [0.25, 0.5, 0.75, 1.0].forEach(ring => {{
    const pts = angles.map(a => polar(a, R*ring).join(',')).join(' ');
    svg += `<polygon points="${{pts}}" fill="none" stroke="{BORDER}" stroke-width="1"/>`;
  }});

  // Axes
  angles.forEach((a, i) => {{
    const [px,py] = polar(a, R);
    const [lx,ly] = polar(a, R+30);
    svg += `<line x1="${{cx}}" y1="${{cy}}" x2="${{px}}" y2="${{py}}" stroke="{BORDER}" stroke-width="1"/>`;
    const anchor = lx < cx-10 ? 'end' : lx > cx+10 ? 'start' : 'middle';
    svg += `<text x="${{lx}}" y="${{ly+4}}" text-anchor="${{anchor}}" fill="{TEXT}" font-size="12" font-weight="500">${{labels[i]}}</text>`;
  }});

  // Data
  filtered.forEach((run, ri) => {{
    const color = COLORS[ALL_RUNS.findIndex(r => r.id === run.id) % COLORS.length];
    const vals = metrics.map(k => k === '_speed' ? 1 - run.avg_latency_ms / maxLat : (run[k] || 0));
    const pts = vals.map((v, i) => polar(angles[i], R*v).join(',')).join(' ');
    svg += `<polygon points="${{pts}}" fill="${{color}}" fill-opacity="0.12" stroke="${{color}}" stroke-width="2"/>`;
    vals.forEach((v, i) => {{
      const [dx,dy] = polar(angles[i], R*v);
      svg += `<circle cx="${{dx}}" cy="${{dy}}" r="3.5" fill="${{color}}" stroke="{BG}" stroke-width="1"/>`;
    }});
  }});

  svg += '</svg>';
  container.innerHTML = svg;
}}

// ── Details Panel ──
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

  let html = `<h3 style="margin-bottom:1rem">${{run.model}} \u2014 ${{run.items.length}} items</h3>`;
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
          <span class="badge metric">exact: ${{item.exact_match ? '\u2713' : '\u2717'}}</span>
          <span class="badge metric">contains: ${{item.contains ? '\u2713' : '\u2717'}}</span>
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

    radar = _svg_radar_chart(runs)
    (out / "radar.svg").write_text(radar)
    print(f"  \u2192 {out / 'radar.svg'}")

    bar = _svg_bar_chart(runs)
    (out / "comparison.svg").write_text(bar)
    print(f"  \u2192 {out / 'comparison.svg'}")

    latency = _svg_latency_chart(runs)
    (out / "latency.svg").write_text(latency)
    print(f"  \u2192 {out / 'latency.svg'}")

    html = _build_html(runs)
    (out / "index.html").write_text(html)
    print(f"  \u2192 {out / 'index.html'}")

    print("Done.")


if __name__ == "__main__":
    db = sys.argv[1] if len(sys.argv) > 1 else "results.db"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs"
    generate(db, out)
