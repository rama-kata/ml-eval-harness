"""Generate an interactive ECharts HTML dashboard from eval results."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from textwrap import dedent


# -- Dark theme tokens (GitHub-style) --
BG = "#0d1117"
BG_CARD = "#161b22"
BORDER = "#30363d"
TEXT = "#e6edf3"
TEXT_DIM = "#8b949e"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
YELLOW = "#d29922"
RED = "#f85149"
PURPLE = "#bc8cff"
CYAN = "#39d2c0"
PINK = "#f778ba"
ACCENT_COLORS = [ACCENT, GREEN, YELLOW, RED, PURPLE, CYAN, PINK]

ECHARTS_CDN = "https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"


def load_runs(db_path: str) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, model, dataset, accuracy, contains_rate, avg_token_f1, "
        "avg_latency_ms, p95_latency_ms, judge_correct_rate, judge_partial_rate, "
        "raw_results, created_at FROM runs ORDER BY id"
    ).fetchall()
    return [dict(r) for r in rows]


def _summary_cards(runs: list[dict]) -> str:
    """Build HTML for the summary stat cards at the top."""
    total = len(runs)
    models = list({r["model"] for r in runs})

    best_acc = max(runs, key=lambda r: r.get("accuracy") or 0)
    fastest = min(runs, key=lambda r: r.get("avg_latency_ms") or float("inf"))
    best_f1 = max(runs, key=lambda r: r.get("avg_token_f1") or 0)

    cards = [
        ("Total Runs", str(total), f"{len(models)} model{'s' if len(models) != 1 else ''}"),
        ("Best Exact Match", f"{(best_acc['accuracy'] or 0):.1%}", best_acc["model"]),
        ("Best Token F1", f"{(best_f1['avg_token_f1'] or 0):.3f}", best_f1["model"]),
        ("Fastest", f"{(fastest['avg_latency_ms'] or 0):.0f}ms", fastest["model"]),
    ]

    html = '<div class="summary-cards">\n'
    for label, value, sub in cards:
        html += dedent(f"""\
            <div class="card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
                <div class="card-sub">{sub}</div>
            </div>\n""")
    html += "</div>\n"
    return html


def _chart_data(runs: list[dict]) -> dict:
    """Prepare all chart data as a single JSON-serializable dict."""
    models = [r["model"] for r in runs]
    has_judge = any(r.get("judge_correct_rate") is not None for r in runs)

    # Metrics for heatmap and bar chart
    metric_keys = ["accuracy", "contains_rate", "avg_token_f1"]
    metric_labels = ["Exact Match", "Contains", "Token F1"]
    if has_judge:
        metric_keys.append("judge_correct_rate")
        metric_labels.append("Judge Correct")

    # Heatmap: [col_index, row_index, value]
    heatmap_data = []
    for mi, key in enumerate(metric_keys):
        for ri, run in enumerate(runs):
            val = run.get(key) or 0
            heatmap_data.append([mi, ri, round(val, 3)])

    # Bar chart series
    bar_series = []
    for mi, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        bar_series.append({
            "name": label,
            "type": "bar",
            "data": [round((r.get(key) or 0) * 100, 1) for r in runs],
            "itemStyle": {"borderRadius": [4, 4, 0, 0]},
        })

    # Latency: avg bars + p95 error bars
    avg_latencies = [round(r.get("avg_latency_ms") or 0, 1) for r in runs]
    p95_latencies = [round(r.get("p95_latency_ms") or r.get("avg_latency_ms") or 0, 1)
                     for r in runs]
    latency_error = [[0, round(p95 - avg, 1)] for avg, p95 in zip(avg_latencies, p95_latencies)]

    # Scatter: latency vs accuracy
    scatter_data = []
    for run in runs:
        scatter_data.append({
            "value": [
                round(run.get("avg_latency_ms") or 0, 1),
                round((run.get("accuracy") or 0) * 100, 1),
            ],
            "name": run["model"],
        })

    # Per-item details
    details = {}
    for run in runs:
        raw = json.loads(run["raw_results"]) if run.get("raw_results") else {}
        details[run["id"]] = {
            "model": run["model"],
            "dataset": run["dataset"],
            "items": raw.get("details", []),
        }

    return {
        "models": models,
        "metricLabels": metric_labels,
        "heatmapData": heatmap_data,
        "barSeries": bar_series,
        "avgLatencies": avg_latencies,
        "p95Latencies": p95_latencies,
        "latencyError": latency_error,
        "scatterData": scatter_data,
        "details": details,
        "runIds": [r["id"] for r in runs],
        "createdAt": [r["created_at"][:19] for r in runs],
        "hasJudge": has_judge,
    }


def _build_html(runs: list[dict]) -> str:
    """Build the full self-contained ECharts HTML dashboard."""
    summary = _summary_cards(runs)
    data = _chart_data(runs)
    data_json = json.dumps(data, indent=None)

    # Table rows
    table_rows = ""
    for run in runs:
        judge_col = (
            f'<td>{run["judge_correct_rate"]:.1%}</td>'
            if run.get("judge_correct_rate") is not None
            else '<td class="dim">\u2014</td>'
        )
        table_rows += dedent(f"""\
            <tr data-run-id="{run['id']}" onclick="showDetails({run['id']})">
                <td class="model-name">{run['model']}</td>
                <td>{(run['accuracy'] or 0):.1%}</td>
                <td>{(run['contains_rate'] or 0):.1%}</td>
                <td>{(run.get('avg_token_f1') or 0):.3f}</td>
                <td>{(run.get('avg_latency_ms') or 0):.0f}ms</td>
                {judge_col}
                <td class="dim">{run['created_at'][:19]}</td>
            </tr>\n""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ml-eval-harness \u2014 Results Dashboard</title>
<script src="{ECHARTS_CDN}"></script>
<style>
  :root {{
    --bg: {BG}; --card: {BG_CARD}; --border: {BORDER};
    --text: {TEXT}; --dim: {TEXT_DIM};
    --blue: {ACCENT}; --green: {GREEN}; --yellow: {YELLOW};
    --red: {RED}; --purple: {PURPLE};
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    font-size: 14px; line-height: 1.5;
    padding: 2rem; max-width: 1400px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.8rem; font-weight: 600; margin-bottom: 0.5rem; }}
  h1 span {{ color: var(--blue); }}
  .subtitle {{ color: var(--dim); margin-bottom: 1.5rem; }}

  /* Summary cards */
  .summary-cards {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 2rem;
  }}
  .card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 1.25rem; text-align: center;
  }}
  .card-label {{ color: var(--dim); font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.25rem; }}
  .card-value {{ font-size: 1.8rem; font-weight: 700; color: var(--blue); }}
  .card-sub {{ color: var(--dim); font-size: 12px; margin-top: 0.25rem; }}

  /* Chart grid */
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .chart-card {{
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 1rem; overflow: hidden;
  }}
  .chart-card.full {{ grid-column: 1 / -1; }}
  .chart-container {{ width: 100%; height: 400px; }}
  .chart-container.tall {{ height: 480px; }}

  /* Table */
  .section-title {{
    font-size: 1.1rem; font-weight: 600; margin: 2rem 0 1rem;
    padding-bottom: 0.5rem; border-bottom: 1px solid var(--border);
  }}
  table {{
    width: 100%; border-collapse: collapse; background: var(--card);
    border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
  }}
  th {{
    text-align: left; padding: 12px 16px; font-weight: 600; font-size: 12px;
    text-transform: uppercase; letter-spacing: 0.05em; color: var(--dim);
    border-bottom: 1px solid var(--border); background: var(--card); cursor: pointer;
    user-select: none;
  }}
  th:hover {{ color: var(--text); }}
  th .sort-arrow {{ margin-left: 4px; font-size: 10px; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid var(--border); }}
  tr {{ cursor: pointer; transition: background 0.15s; }}
  tr:hover {{ background: rgba(88,166,255,0.06); }}
  tr.active {{ background: rgba(88,166,255,0.1); }}
  .model-name {{ font-weight: 600; color: var(--blue); }}
  .dim {{ color: var(--dim); }}

  /* Details panel */
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
  .detail-item .response {{
    margin: 0.5rem 0; padding: 0.5rem; background: var(--card); border-radius: 4px;
    word-break: break-word;
  }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 600; margin-right: 4px;
  }}
  .badge.correct {{ background: rgba(63,185,80,0.15); color: var(--green); }}
  .badge.partial {{ background: rgba(210,153,34,0.15); color: var(--yellow); }}
  .badge.wrong {{ background: rgba(248,81,73,0.15); color: var(--red); }}
  .badge.metric {{ background: rgba(88,166,255,0.1); color: var(--blue); }}
  .f1-bar {{ height: 4px; border-radius: 2px; background: var(--border); margin-top: 4px; overflow: hidden; }}
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

{summary}

<div class="charts">
  <div class="chart-card">
    <div id="chart-heatmap" class="chart-container"></div>
  </div>
  <div class="chart-card">
    <div id="chart-scatter" class="chart-container"></div>
  </div>
  <div class="chart-card full">
    <div id="chart-bars" class="chart-container"></div>
  </div>
  <div class="chart-card full">
    <div id="chart-latency" class="chart-container tall"></div>
  </div>
</div>

<div class="section-title">All Runs</div>
<table id="runs-table">
  <thead>
    <tr>
      <th data-col="model">Model <span class="sort-arrow"></span></th>
      <th data-col="accuracy">Exact Match <span class="sort-arrow"></span></th>
      <th data-col="contains">Contains <span class="sort-arrow"></span></th>
      <th data-col="f1">Token F1 <span class="sort-arrow"></span></th>
      <th data-col="latency">Avg Latency <span class="sort-arrow"></span></th>
      <th data-col="judge">Judge <span class="sort-arrow"></span></th>
      <th data-col="date">Date <span class="sort-arrow"></span></th>
    </tr>
  </thead>
  <tbody>
    {table_rows}
  </tbody>
</table>

<div class="section-title">Per-Item Details <span class="dim" style="font-weight:400;font-size:0.85rem">(click a run above)</span></div>
<div id="details-panel"></div>

<script>
const D = {data_json};
const COLORS = {json.dumps(ACCENT_COLORS)};

// -- ECharts dark theme base --
const THEME = {{
  backgroundColor: 'transparent',
  textStyle: {{ color: '{TEXT}' }},
  legend: {{ textStyle: {{ color: '{TEXT_DIM}' }} }},
  categoryAxis: {{
    axisLine: {{ lineStyle: {{ color: '{BORDER}' }} }},
    axisLabel: {{ color: '{TEXT_DIM}' }},
    splitLine: {{ lineStyle: {{ color: '{BORDER}', type: 'dashed' }} }},
  }},
  valueAxis: {{
    axisLine: {{ lineStyle: {{ color: '{BORDER}' }} }},
    axisLabel: {{ color: '{TEXT_DIM}' }},
    splitLine: {{ lineStyle: {{ color: '{BORDER}', type: 'dashed' }} }},
  }},
}};

function initChart(id) {{
  const el = document.getElementById(id);
  const chart = echarts.init(el, null, {{ renderer: 'canvas' }});
  window.addEventListener('resize', () => chart.resize());
  return chart;
}}

// -- Heatmap --
(function() {{
  const chart = initChart('chart-heatmap');
  const maxVal = Math.max(...D.heatmapData.map(d => d[2]), 0.01);
  chart.setOption({{
    ...THEME,
    title: {{ text: 'Metric Heatmap', left: 'center', top: 10,
             textStyle: {{ color: '{TEXT}', fontSize: 16, fontWeight: 600 }} }},
    tooltip: {{
      formatter: function(p) {{
        const metric = D.metricLabels[p.data[0]];
        const model = D.models[p.data[1]];
        return '<b>' + model + '</b><br/>' + metric + ': ' + (p.data[2] * 100).toFixed(1) + '%';
      }}
    }},
    grid: {{ top: 50, bottom: 60, left: 160, right: 40 }},
    xAxis: {{
      type: 'category', data: D.metricLabels, position: 'bottom',
      axisLabel: {{ color: '{TEXT_DIM}', fontSize: 12 }},
      axisLine: {{ lineStyle: {{ color: '{BORDER}' }} }},
    }},
    yAxis: {{
      type: 'category', data: D.models,
      axisLabel: {{ color: '{TEXT}', fontSize: 12 }},
      axisLine: {{ lineStyle: {{ color: '{BORDER}' }} }},
    }},
    visualMap: {{
      min: 0, max: 1, calculable: false, orient: 'horizontal',
      left: 'center', bottom: 5,
      inRange: {{ color: ['{BG}', '{ACCENT}', '{GREEN}'] }},
      textStyle: {{ color: '{TEXT_DIM}' }},
      formatter: function(v) {{ return (v * 100).toFixed(0) + '%'; }},
    }},
    series: [{{
      type: 'heatmap', data: D.heatmapData,
      label: {{
        show: true,
        formatter: function(p) {{ return (p.data[2] * 100).toFixed(1) + '%'; }},
        color: '{TEXT}', fontSize: 12, fontWeight: 600,
      }},
      itemStyle: {{ borderRadius: 4, borderColor: '{BG_CARD}', borderWidth: 3 }},
      emphasis: {{ itemStyle: {{ shadowBlur: 10, shadowColor: 'rgba(88,166,255,0.4)' }} }},
    }}],
    animationDuration: 800,
    animationEasing: 'cubicOut',
  }});
}})();

// -- Scatter: Latency vs Accuracy --
(function() {{
  const chart = initChart('chart-scatter');
  chart.setOption({{
    ...THEME,
    title: {{ text: 'Latency vs Accuracy', left: 'center', top: 10,
             textStyle: {{ color: '{TEXT}', fontSize: 16, fontWeight: 600 }} }},
    tooltip: {{
      formatter: function(p) {{
        return '<b>' + p.data.name + '</b><br/>'
          + 'Latency: ' + p.data.value[0] + 'ms<br/>'
          + 'Accuracy: ' + p.data.value[1] + '%';
      }}
    }},
    grid: {{ top: 50, bottom: 50, left: 70, right: 40 }},
    xAxis: {{
      name: 'Avg Latency (ms)', nameLocation: 'center', nameGap: 30,
      nameTextStyle: {{ color: '{TEXT_DIM}' }},
      type: 'value',
      axisLabel: {{ color: '{TEXT_DIM}' }},
      axisLine: {{ lineStyle: {{ color: '{BORDER}' }} }},
      splitLine: {{ lineStyle: {{ color: '{BORDER}', type: 'dashed' }} }},
    }},
    yAxis: {{
      name: 'Exact Match (%)', nameLocation: 'center', nameGap: 45,
      nameTextStyle: {{ color: '{TEXT_DIM}' }},
      type: 'value', min: 0, max: 100,
      axisLabel: {{ color: '{TEXT_DIM}' }},
      axisLine: {{ lineStyle: {{ color: '{BORDER}' }} }},
      splitLine: {{ lineStyle: {{ color: '{BORDER}', type: 'dashed' }} }},
    }},
    series: [{{
      type: 'scatter', symbolSize: 18,
      data: D.scatterData,
      itemStyle: {{ color: '{ACCENT}', borderColor: '{TEXT}', borderWidth: 1 }},
      emphasis: {{
        itemStyle: {{ shadowBlur: 15, shadowColor: 'rgba(88,166,255,0.5)' }},
        scale: 1.4,
      }},
      label: {{
        show: true, position: 'top',
        formatter: function(p) {{ return p.data.name; }},
        color: '{TEXT_DIM}', fontSize: 11,
      }},
    }}],
    animationDuration: 1000,
    animationEasing: 'elasticOut',
  }});
}})();

// -- Grouped Bar Chart --
(function() {{
  const chart = initChart('chart-bars');
  const series = D.barSeries.map(function(s, i) {{
    s.itemStyle = s.itemStyle || {{}};
    s.itemStyle.color = COLORS[i % COLORS.length];
    s.emphasis = {{ itemStyle: {{ shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' }} }};
    return s;
  }});
  chart.setOption({{
    ...THEME,
    title: {{ text: 'Model Comparison \u2014 Accuracy Metrics', left: 'center', top: 10,
             textStyle: {{ color: '{TEXT}', fontSize: 16, fontWeight: 600 }} }},
    tooltip: {{
      trigger: 'axis', axisPointer: {{ type: 'shadow' }},
      formatter: function(params) {{
        let html = '<b>' + params[0].axisValue + '</b><br/>';
        params.forEach(function(p) {{
          html += '<span style="color:' + p.color + '">\u25cf</span> '
            + p.seriesName + ': ' + p.data + '%<br/>';
        }});
        return html;
      }}
    }},
    legend: {{
      top: 38, textStyle: {{ color: '{TEXT_DIM}' }},
    }},
    grid: {{ top: 80, bottom: 60, left: 50, right: 30 }},
    xAxis: {{
      type: 'category', data: D.models,
      axisLabel: {{ color: '{TEXT_DIM}', fontSize: 11, rotate: D.models.length > 5 ? 20 : 0 }},
      axisLine: {{ lineStyle: {{ color: '{BORDER}' }} }},
    }},
    yAxis: {{
      type: 'value', max: 100,
      axisLabel: {{ color: '{TEXT_DIM}', formatter: '{{value}}%' }},
      axisLine: {{ lineStyle: {{ color: '{BORDER}' }} }},
      splitLine: {{ lineStyle: {{ color: '{BORDER}', type: 'dashed' }} }},
    }},
    series: series,
    animationDuration: 800,
    animationEasing: 'cubicOut',
    animationDelay: function(idx) {{ return idx * 80; }},
  }});
}})();

// -- Latency Chart --
(function() {{
  const chart = initChart('chart-latency');
  chart.setOption({{
    ...THEME,
    title: {{ text: 'Latency Comparison (avg + p95)', left: 'center', top: 10,
             textStyle: {{ color: '{TEXT}', fontSize: 16, fontWeight: 600 }} }},
    tooltip: {{
      trigger: 'axis', axisPointer: {{ type: 'shadow' }},
      formatter: function(params) {{
        const idx = params[0].dataIndex;
        return '<b>' + D.models[idx] + '</b><br/>'
          + 'Avg: ' + D.avgLatencies[idx] + 'ms<br/>'
          + 'P95: ' + D.p95Latencies[idx] + 'ms';
      }}
    }},
    legend: {{
      top: 38, textStyle: {{ color: '{TEXT_DIM}' }},
    }},
    grid: {{ top: 80, bottom: 60, left: 70, right: 30 }},
    xAxis: {{
      type: 'category', data: D.models,
      axisLabel: {{ color: '{TEXT_DIM}', fontSize: 11, rotate: D.models.length > 5 ? 20 : 0 }},
      axisLine: {{ lineStyle: {{ color: '{BORDER}' }} }},
    }},
    yAxis: {{
      type: 'value',
      axisLabel: {{ color: '{TEXT_DIM}', formatter: '{{value}}ms' }},
      axisLine: {{ lineStyle: {{ color: '{BORDER}' }} }},
      splitLine: {{ lineStyle: {{ color: '{BORDER}', type: 'dashed' }} }},
    }},
    series: [
      {{
        name: 'Avg Latency',
        type: 'bar',
        data: D.avgLatencies,
        itemStyle: {{
          borderRadius: [4, 4, 0, 0],
          color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
            {{ offset: 0, color: '{ACCENT}' }},
            {{ offset: 1, color: 'rgba(88,166,255,0.3)' }},
          ]),
        }},
        emphasis: {{ itemStyle: {{ shadowBlur: 10, shadowColor: 'rgba(88,166,255,0.4)' }} }},
        label: {{
          show: true, position: 'top',
          formatter: function(p) {{ return p.value + 'ms'; }},
          color: '{TEXT_DIM}', fontSize: 11,
        }},
      }},
      {{
        name: 'P95 Latency',
        type: 'bar',
        data: D.p95Latencies,
        itemStyle: {{
          borderRadius: [4, 4, 0, 0],
          color: 'rgba(88,166,255,0.15)',
          borderColor: '{ACCENT}',
          borderWidth: 1,
          borderType: 'dashed',
        }},
        barGap: '-100%',
        z: -1,
      }},
    ],
    animationDuration: 800,
    animationEasing: 'cubicOut',
    animationDelay: function(idx) {{ return idx * 100; }},
  }});
}})();

// -- Table sorting --
(function() {{
  const table = document.getElementById('runs-table');
  const tbody = table.querySelector('tbody');
  const headers = table.querySelectorAll('th');
  let sortCol = null;
  let sortAsc = true;

  headers.forEach(function(th) {{
    th.addEventListener('click', function() {{
      const col = th.dataset.col;
      if (sortCol === col) {{ sortAsc = !sortAsc; }}
      else {{ sortCol = col; sortAsc = true; }}

      headers.forEach(function(h) {{ h.querySelector('.sort-arrow').textContent = ''; }});
      th.querySelector('.sort-arrow').textContent = sortAsc ? '\u25b2' : '\u25bc';

      const rows = Array.from(tbody.querySelectorAll('tr'));
      rows.sort(function(a, b) {{
        const cellsA = a.querySelectorAll('td');
        const cellsB = b.querySelectorAll('td');
        let idx = Array.from(headers).indexOf(th);
        let va = cellsA[idx].textContent.trim();
        let vb = cellsB[idx].textContent.trim();

        // Try numeric sort
        const na = parseFloat(va);
        const nb = parseFloat(vb);
        if (!isNaN(na) && !isNaN(nb)) {{
          return sortAsc ? na - nb : nb - na;
        }}
        return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      }});
      rows.forEach(function(r) {{ tbody.appendChild(r); }});
    }});
  }});
}})();

// -- Per-item details --
function showDetails(runId) {{
  const panel = document.getElementById('details-panel');
  const run = D.details[runId];
  if (!run || !run.items.length) {{
    panel.innerHTML = '<p class="dim">No per-item details for this run.</p>';
    panel.classList.add('visible');
    return;
  }}

  document.querySelectorAll('tr.active').forEach(function(r) {{ r.classList.remove('active'); }});
  const row = document.querySelector('tr[data-run-id="' + runId + '"]');
  if (row) row.classList.add('active');

  let html = '<h3 style="margin-bottom:1rem">' + run.model + ' \u2014 ' + run.items.length + ' items</h3>';
  run.items.forEach(function(item, i) {{
    const judgeBadge = item.judge_verdict
      ? '<span class="badge ' + item.judge_verdict + '">' + item.judge_verdict + '</span>'
        + ' <span class="dim" style="font-size:12px">' + (item.judge_reason || '') + '</span>'
      : '';

    const f1Pct = ((item.token_f1 || 0) * 100).toFixed(0);

    html += '<div class="detail-item">'
      + '<div class="prompt">' + (i+1) + '. ' + item.prompt + '</div>'
      + '<div><strong>Expected:</strong> ' + item.expected + '</div>'
      + '<div class="response"><strong>Response:</strong> ' + item.response + '</div>'
      + '<div>'
      + '<span class="badge metric">exact: ' + (item.exact_match ? '\u2713' : '\u2717') + '</span>'
      + '<span class="badge metric">contains: ' + (item.contains ? '\u2713' : '\u2717') + '</span>'
      + '<span class="badge metric">F1: ' + f1Pct + '%</span>'
      + '<span class="badge metric">' + (item.latency_s * 1000).toFixed(0) + 'ms</span>'
      + ' ' + judgeBadge
      + '</div>'
      + '<div class="f1-bar"><div class="f1-bar-fill" style="width:' + f1Pct + '%"></div></div>'
      + '</div>';
  }});

  panel.innerHTML = html;
  panel.classList.add('visible');
}}
</script>
</body>
</html>"""


# ── Static SVGs for README ───────────────────────────────────────────────────


def _svg_heatmap(runs: list[dict], width: int = 820, height: int = 0) -> str:
    """Static SVG heatmap: models (rows) x metrics (cols), color-coded cells."""
    metrics = [("accuracy", "Exact Match"), ("contains_rate", "Contains"),
               ("avg_token_f1", "Token F1")]
    has_judge = any(r.get("judge_correct_rate") is not None for r in runs)
    if has_judge:
        metrics.append(("judge_correct_rate", "Judge"))

    n_models = len(runs)
    n_metrics = len(metrics)
    label_w = 180
    cell_w = (width - label_w - 20) // n_metrics
    cell_h = 38
    header_h = 50
    title_h = 40
    padding = 20
    if height == 0:
        height = title_h + header_h + n_models * cell_h + padding

    def lerp_color(t: float) -> str:
        """Interpolate from dim red through yellow to green."""
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            r = int(248 - (248 - 210) * (t / 0.5))
            g = int(81 + (153 - 81) * (t / 0.5))
            b = int(73 - (73 - 34) * (t / 0.5))
        else:
            s = (t - 0.5) / 0.5
            r = int(210 - (210 - 63) * s)
            g = int(153 + (185 - 153) * s)
            b = int(34 + (80 - 34) * s)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" rx="8" fill="{BG}"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" fill="{TEXT}" '
        f'font-size="15" font-weight="600" font-family="sans-serif">Metric Heatmap</text>',
    ]

    # Column headers
    for mi, (_, label) in enumerate(metrics):
        cx = label_w + mi * cell_w + cell_w / 2
        parts.append(
            f'<text x="{cx}" y="{title_h + 18}" text-anchor="middle" fill="{TEXT_DIM}" '
            f'font-size="12" font-family="sans-serif" font-weight="600">{label}</text>'
        )

    # Rows
    for ri, run in enumerate(runs):
        y = title_h + header_h + ri * cell_h
        # Model label
        parts.append(
            f'<text x="{label_w - 10}" y="{y + cell_h / 2 + 5}" text-anchor="end" fill="{TEXT}" '
            f'font-size="12" font-family="sans-serif">{run["model"]}</text>'
        )
        for mi, (key, _) in enumerate(metrics):
            val = run.get(key) or 0
            cx = label_w + mi * cell_w
            color = lerp_color(val)
            parts.append(
                f'<rect x="{cx + 2}" y="{y + 2}" width="{cell_w - 4}" height="{cell_h - 4}" '
                f'rx="4" fill="{color}" opacity="0.85"/>'
            )
            text_color = TEXT if val < 0.65 else BG
            parts.append(
                f'<text x="{cx + cell_w / 2}" y="{y + cell_h / 2 + 5}" text-anchor="middle" '
                f'fill="{text_color}" font-size="12" font-weight="600" '
                f'font-family="sans-serif">{val:.1%}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def _svg_latency(runs: list[dict], width: int = 820, height: int = 320) -> str:
    """Static SVG bar chart for latency comparison with p95 markers."""
    margin = {"top": 55, "right": 30, "bottom": 70, "left": 80}
    chart_w = width - margin["left"] - margin["right"]
    chart_h = height - margin["top"] - margin["bottom"]

    # Scale to max of p95 so everything fits
    max_val = max(
        max((r.get("p95_latency_ms") or r.get("avg_latency_ms") or 0) for r in runs),
        1,
    ) * 1.15
    bar_w = min(chart_w / (len(runs) * 1.8), 70)
    total_bars_w = bar_w * len(runs)
    gap = (chart_w - total_bars_w) / (len(runs) + 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" rx="8" fill="{BG}"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" fill="{TEXT}" '
        f'font-size="15" font-weight="600" font-family="sans-serif">'
        f'Latency Comparison (avg + p95)</text>',
    ]

    # Y gridlines
    for i in range(5):
        val = max_val * i / 4
        y = margin["top"] + chart_h - (val / max_val * chart_h)
        parts.append(
            f'<line x1="{margin["left"]}" y1="{y}" x2="{width - margin["right"]}" '
            f'y2="{y}" stroke="{BORDER}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin["left"] - 8}" y="{y + 4}" text-anchor="end" '
            f'fill="{TEXT_DIM}" font-size="11" font-family="sans-serif">{val:.0f}ms</text>'
        )

    # Bars
    for i, run in enumerate(runs):
        color = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        avg = run.get("avg_latency_ms") or 0
        p95 = run.get("p95_latency_ms") or avg
        x = margin["left"] + gap + i * (bar_w + gap)
        cx = x + bar_w / 2

        # Avg bar
        bar_h = (avg / max_val) * chart_h
        y = margin["top"] + chart_h - bar_h
        parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w}" height="{bar_h}" '
            f'rx="3" fill="{color}" opacity="0.85"/>'
        )

        # P95 whisker
        p95_h = (p95 / max_val) * chart_h
        p95_y = margin["top"] + chart_h - p95_h
        parts.append(
            f'<line x1="{cx}" y1="{p95_y}" x2="{cx}" y2="{y}" '
            f'stroke="{color}" stroke-width="2" opacity="0.5"/>'
        )
        parts.append(
            f'<line x1="{cx - 8}" y1="{p95_y}" x2="{cx + 8}" y2="{p95_y}" '
            f'stroke="{color}" stroke-width="2" opacity="0.5"/>'
        )

        # Value label
        parts.append(
            f'<text x="{cx}" y="{y - 6}" text-anchor="middle" fill="{TEXT}" '
            f'font-size="11" font-weight="600" font-family="sans-serif">{avg:.0f}ms</text>'
        )

        # Model name (rotated if many)
        label_y = margin["top"] + chart_h + 16
        if len(runs) > 5:
            parts.append(
                f'<text x="{cx}" y="{label_y}" text-anchor="end" fill="{TEXT_DIM}" '
                f'font-size="11" font-family="sans-serif" '
                f'transform="rotate(-30 {cx} {label_y})">{run["model"]}</text>'
            )
        else:
            parts.append(
                f'<text x="{cx}" y="{label_y}" text-anchor="middle" fill="{TEXT_DIM}" '
                f'font-size="11" font-family="sans-serif">{run["model"]}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def generate(db_path: str = "results.db", output_dir: str = "docs"):
    """Generate the ECharts HTML dashboard and static SVGs for README."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(db_path)
    if not runs:
        print("No runs found in database.")
        return

    print(f"Generating dashboard for {len(runs)} run(s)...")

    # Interactive HTML dashboard
    html = _build_html(runs)
    (out / "index.html").write_text(html)
    print(f"  \u2192 {out / 'index.html'}")

    # Static SVGs for README
    heatmap = _svg_heatmap(runs)
    (out / "heatmap.svg").write_text(heatmap)
    print(f"  \u2192 {out / 'heatmap.svg'}")

    latency = _svg_latency(runs)
    (out / "latency.svg").write_text(latency)
    print(f"  \u2192 {out / 'latency.svg'}")

    # Keep .nojekyll for GitHub Pages
    (out / ".nojekyll").touch()

    print("Done.")


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else "results.db"
    out = sys.argv[2] if len(sys.argv) > 2 else "docs"
    generate(db, out)
