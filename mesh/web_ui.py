# -*- coding: utf-8 -*-
# web_ui.py
# MeshTime Web-UI: Pairwise ΔMesh-Time (via offsets), robust jitter (IQR/MAD), topology, heatmap

import json
import math
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template_string, jsonify

BASE_DIR = Path(__file__).resolve().parent.parent   # /home/pi/mesh_time
DB_PATH  = BASE_DIR / "mesh_data.sqlite"
CFG_PATH = BASE_DIR / "config" / "nodes.json"

from mesh.storage import Storage  # ensures schema

app = Flask(__name__)


# ------------------------------------------------------------
# DB / Config helpers
# ------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_config():
    if not CFG_PATH.exists():
        return {}
    with CFG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_topology():
    """
    Reads config/nodes.json and returns a simple mesh topology:
      - nodes: [{id, ip, color, is_root}]
      - links: [{source, target}]
    """
    cfg = load_config()
    nodes = []
    links = []

    for node_id, entry in cfg.items():
        if node_id == "sync":
            continue

        ip = entry.get("ip")
        color = entry.get("color") or "#3498db"
        sync_cfg = entry.get("sync", {}) or {}
        is_root = bool(sync_cfg.get("is_root", False))

        nodes.append({"id": node_id, "ip": ip, "color": color, "is_root": is_root})

        for neigh in entry.get("neighbors", []):
            if node_id < neigh:
                links.append({"source": node_id, "target": neigh})

    return {"nodes": nodes, "links": links}


# ------------------------------------------------------------
# Robust stats
# ------------------------------------------------------------

def _quantile_sorted(xs, p):
    n = len(xs)
    if n == 0:
        return None
    pos = p * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def robust_sigma_iqr(values):
    """
    Robust sigma estimate via IQR:
        sigma ~= 0.7413 * IQR
    Works well for jitter-like distributions (robust to outliers + drift).
    """
    if values is None:
        return None
    xs = sorted([float(v) for v in values if v is not None and math.isfinite(float(v))])
    if len(xs) < 8:
        return None
    q25 = _quantile_sorted(xs, 0.25)
    q75 = _quantile_sorted(xs, 0.75)
    if q25 is None or q75 is None:
        return None
    iqr = q75 - q25
    return 0.7413 * iqr


def robust_sigma_mad(values):
    """
    Robust sigma estimate via MAD:
        sigma ~= 1.4826 * MAD
    """
    if values is None:
        return None
    xs = sorted([float(v) for v in values if v is not None and math.isfinite(float(v))])
    if len(xs) < 8:
        return None
    med = _quantile_sorted(xs, 0.50)
    if med is None:
        return None
    abs_dev = sorted([abs(x - med) for x in xs])
    mad = _quantile_sorted(abs_dev, 0.50)
    return 1.4826 * mad if mad is not None else None


# ------------------------------------------------------------
# Data aggregation for charts
# ------------------------------------------------------------

def get_ntp_timeseries(window_seconds=600, max_points=2000):
    """
    Pull ntp_reference rows and build:
      - series: per node [{t_wall, offset_ms, delta_offset_ms}]
      - pairs:  pairwise ΔMesh-Time (via offsets) [{t_wall, delta_ms}]
      - jitter_sigma_pairs: robust jitter per pair (IQR-based sigma)
      - heatmap: binned avg |Δ| per pair/timebin
    """
    conn = get_conn()
    cur = conn.cursor()

    cutoff = time.time() - float(window_seconds)

    rows = cur.execute(
        """
        SELECT node_id, t_wall, offset, created_at
        FROM ntp_reference
        WHERE created_at >= ?
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (cutoff, int(max_points)),
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "series": {},
            "pairs": {},
            "jitter_sigma_pairs": {},
            "jitter_mad_pairs": {},
            "heatmap": {"data": [], "n_bins": 0},
        }

    # group by node
    per_node = {}
    for r in rows:
        per_node.setdefault(r["node_id"], []).append(r)

    # per-node series: offset + delta_offset
    series = {}
    for node_id, node_rows in per_node.items():
        node_rows_sorted = sorted(node_rows, key=lambda r: float(r["t_wall"]))
        out = []
        prev_offset_ms = None
        for r in node_rows_sorted:
            t_wall = float(r["t_wall"])
            offset_ms = float(r["offset"]) * 1000.0
            delta_offset_ms = 0.0 if prev_offset_ms is None else (offset_ms - prev_offset_ms)
            prev_offset_ms = offset_ms
            out.append({"t_wall": t_wall, "offset_ms": offset_ms, "delta_offset_ms": delta_offset_ms})
        series[node_id] = out

    # pairwise ΔMesh-Time:
    # Mesh time = t_wall + offset, so for same t_wall: ΔMesh = offset_a - offset_b
    all_rows_sorted = sorted(rows, key=lambda r: float(r["t_wall"]))
    last_offset_ms = {}  # node_id -> last offset_ms
    pair_series = {}     # "A-B" -> [{t_wall, delta_ms}, ...]

    def norm_pair(a, b):
        return "-".join(sorted([a, b]))

    for r in all_rows_sorted:
        node = r["node_id"]
        t_wall = float(r["t_wall"])
        off_ms = float(r["offset"]) * 1000.0
        last_offset_ms[node] = off_ms

        nodes_now = sorted(last_offset_ms.keys())
        if len(nodes_now) < 2:
            continue

        for i in range(len(nodes_now)):
            for j in range(i + 1, len(nodes_now)):
                a = nodes_now[i]
                b = nodes_now[j]
                pair_id = norm_pair(a, b)
                delta_ms = last_offset_ms[a] - last_offset_ms[b]
                pair_series.setdefault(pair_id, []).append({"t_wall": t_wall, "delta_ms": delta_ms})

    # robust jitter per pair (sigma via IQR and via MAD)
    jitter_sigma_pairs = {}
    jitter_mad_pairs = {}
    for pair_id, pts in pair_series.items():
        deltas = [p["delta_ms"] for p in pts]
        s_iqr = robust_sigma_iqr(deltas)
        s_mad = robust_sigma_mad(deltas)
        if s_iqr is not None:
            jitter_sigma_pairs[pair_id] = s_iqr
        if s_mad is not None:
            jitter_mad_pairs[pair_id] = s_mad

    # heatmap: bin by time and pair using avg |Δ|
    heatmap_data = []
    n_bins = 0
    if pair_series:
        all_t = [float(p["t_wall"]) for pts in pair_series.values() for p in pts]
        t_min = min(all_t)
        t_max = max(all_t)

        if t_max <= t_min:
            n_bins = 1
            bin_width = 1.0
        else:
            n_bins = max(1, min(40, int(len(all_t) / 5) or 1))
            bin_width = (t_max - t_min) / n_bins

        accum = {}  # (pair_id, bin_idx) -> [sum_abs_delta, count]
        for pair_id, pts in pair_series.items():
            for p in pts:
                t = float(p["t_wall"])
                if n_bins == 1:
                    idx = 0
                else:
                    rel = (t - t_min) / (t_max - t_min + 1e-9)
                    idx = int(rel * n_bins)
                    if idx >= n_bins:
                        idx = n_bins - 1
                key = (pair_id, idx)
                v = abs(float(p["delta_ms"]))
                if key not in accum:
                    accum[key] = [v, 1]
                else:
                    accum[key][0] += v
                    accum[key][1] += 1

        for (pair_id, idx), (sum_v, cnt) in accum.items():
            t_center = t_min + (idx + 0.5) * (bin_width if n_bins > 1 else 1.0)
            avg = sum_v / max(1, cnt)
            heatmap_data.append({"pair": pair_id, "t_bin": t_center, "value": avg})

    return {
        "series": series,
        "pairs": pair_series,
        "jitter_sigma_pairs": jitter_sigma_pairs,  # robust (IQR)
        "jitter_mad_pairs": jitter_mad_pairs,      # optional (MAD)
        "heatmap": {"data": heatmap_data, "n_bins": n_bins},
    }


def get_last_by_node():
    """
    Latest row per node for the status table.
    We only need t_wall and offset to compute mesh_time.
    """
    conn = get_conn()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT r.*
        FROM ntp_reference r
        JOIN (
          SELECT node_id, MAX(id) AS max_id
          FROM ntp_reference
          GROUP BY node_id
        ) t ON t.node_id = r.node_id AND t.max_id = r.id
        ORDER BY r.node_id
        """
    ).fetchall()
    conn.close()
    return rows


# ------------------------------------------------------------
# Flask routes
# ------------------------------------------------------------

@app.route("/")
def index():
    last_by_node = get_last_by_node()
    topo = get_topology()
    return render_template_string(
        TEMPLATE,
        last_by_node=last_by_node,
        topo=topo,
        db_path=str(DB_PATH),
    )


@app.route("/api/topology")
def api_topology():
    return jsonify(get_topology())


@app.route("/api/ntp_timeseries")
def api_ntp_timeseries():
    return jsonify(get_ntp_timeseries(window_seconds=600, max_points=2000))


@app.template_filter("datetime_utc")
def datetime_utc(ts):
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


# ------------------------------------------------------------
# HTML + JS template
# ------------------------------------------------------------

TEMPLATE = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>MeshTime Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 1.5rem;
      background: #111;
      color: #eee;
    }
    h1, h2, h3 { margin-top: 0; }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, 2fr);
      gap: 1.5rem;
      align-items: flex-start;
    }
    @media (max-width: 1100px) {
      .grid { grid-template-columns: minmax(0, 1fr); }
    }
    .card {
      background: #1b1b1b;
      border-radius: 12px;
      padding: 1rem 1.25rem;
      box-shadow: 0 0 0 1px rgba(255,255,255,0.04);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    th, td {
      padding: 0.35rem 0.5rem;
      text-align: left;
      vertical-align: top;
    }
    th {
      border-bottom: 1px solid rgba(255,255,255,0.15);
      font-weight: 600;
    }
    tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
    .pill {
      display: inline-block;
      padding: 0.1rem 0.5rem;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 500;
    }
    .pill-ok   { background: rgba(46,204,113,0.12); color:#2ecc71; }
    .pill-warn { background: rgba(241,196,15,0.12); color:#f1c40f; }
    .pill-bad  { background: rgba(231,76,60,0.12);  color:#e74c3c; }
    .small { font-size: 0.8rem; opacity: 0.7; }
    canvas { max-width: 100%; }
    #meshCanvas {
      width: 100%;
      height: 260px;
      background: #171717;
      border-radius: 10px;
    }
    .plots-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1rem;
    }
    @media (max-width: 1200px) {
      .plots-grid { grid-template-columns: minmax(0, 1fr); }
    }
    .footer {
      margin-top: 1rem;
      font-size: 0.8rem;
      opacity: 0.6;
    }
    code { color: rgba(255,255,255,0.85); }
  </style>

  <!-- Chart.js + Date adapter + matrix plugin -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@1.3.0/dist/chartjs-chart-matrix.min.js"></script>
</head>

<body>
  <h1>MeshTime Monitor</h1>
  <p class="small">Datenquelle: <code>{{ db_path }}</code></p>

  <div class="grid">
    <div>
      <div class="card">
        <h2>Aktueller Status pro Node</h2>
        {% if last_by_node %}
        <table>
          <thead>
            <tr>
              <th>Node</th>
              <th>Mesh-Time (UTC)</th>
              <th>Offset</th>
              <th>Wallclock (UTC)</th>
            </tr>
          </thead>
          <tbody>
            {% for row in last_by_node %}
              {% set t_wall = row["t_wall"] | float %}
              {% set offset_s = row["offset"] | float %}
              {% set mesh_ts = t_wall + offset_s %}
              {% set offset_ms = offset_s * 1000.0 %}
              {% if offset_ms|abs < 5 %}
                {% set cls = "pill-ok" %}
              {% elif offset_ms|abs < 20 %}
                {% set cls = "pill-warn" %}
              {% else %}
                {% set cls = "pill-bad" %}
              {% endif %}
              <tr>
                <td>{{ row["node_id"] }}</td>
                <td>{{ mesh_ts | datetime_utc }}<div class="small">{{ "%.3f"|format(mesh_ts) }}</div></td>
                <td><span class="pill {{ cls }}">{{ "%.2f"|format(offset_ms) }} ms</span></td>
                <td class="small">{{ t_wall | int | datetime_utc }}</td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
        {% else %}
          <p>Keine ntp_reference-Daten gefunden.</p>
        {% endif %}
      </div>

      <div class="card" style="margin-top:1rem;">
        <h2>Mesh-Topologie</h2>
        <canvas id="meshCanvas"></canvas>
        <p class="small">
          Grün = Node hat Daten, grau = bisher keine. Dicker Rand = <strong>Root</strong> (falls vorhanden).
        </p>
      </div>
    </div>

    <div class="card">
      <h2>Mesh-Time Plots (letzte 10 Minuten)</h2>
      <div class="plots-grid">
        <div>
          <h3 style="font-size:0.95rem;">1) Paarweise ΔMesh-Time (aus Offsets)</h3>
          <canvas id="pairChart" height="160"></canvas>
        </div>
        <div>
          <h3 style="font-size:0.95rem;">2) Offset-Änderung pro Node (Δoffset)</h3>
          <canvas id="offsetDeltaChart" height="160"></canvas>
        </div>
        <div>
          <h3 style="font-size:0.95rem;">3) Jitter der ΔMesh-Time (robust: σ̂ ≈ 0.741·IQR)</h3>
          <canvas id="jitterBarChart" height="160"></canvas>
        </div>
        <div>
          <h3 style="font-size:0.95rem;">4) ΔMesh-Time Heatmap (|Δ| gebinnt)</h3>
          <canvas id="heatmapChart" height="160"></canvas>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    Faustregel: wenn alle ΔMesh-Linien flach bleiben und der Jitter klein ist,
    hast du einen kleinen Zeit-Konsens-Computer gebaut.
  </div>

  <script>
    const colors = [
      'rgba(46, 204, 113, 0.5)',
      'rgba(52, 152, 219, 0.5)',
      'rgba(231, 76, 60, 0.5)',
      'rgba(241, 196, 15, 0.5)',
      'rgba(155, 89, 182, 0.5)'
    ];

    let pairChart, offsetDeltaChart, jitterBarChart, heatmapChart;

    function makeLineChart(ctx, yLabel) {
      return new Chart(ctx, {
        type: 'line',
        data: { datasets: [] },
        options: {
          responsive: true,
          animation: false,
          scales: {
            x: {
              type: 'time',
              time: { unit: 'second' },
              ticks: { color: '#aaa' },
              grid: { color: 'rgba(255,255,255,0.05)' }
            },
            y: {
              ticks: {
                color: '#aaa',
                callback: (v) => (v.toFixed ? v.toFixed(1) : v) + ' ' + yLabel
              },
              grid: { color: 'rgba(255,255,255,0.05)' }
            }
          },
          plugins: {
            legend: { labels: { color: '#eee' } },
            tooltip: {
              callbacks: {
                label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} ${yLabel}`
              }
            }
          },
          elements: {
            line: { tension: 0.1 },
            point: { radius: 0 }
          }
        }
      });
    }

    function makeBarChart(ctx, yLabel) {
      return new Chart(ctx, {
        type: 'bar',
        data: {
          labels: [],
          datasets: [{
            label: 'Jitter (robust σ̂ ≈ 0.741·IQR)',
            data: [],
            backgroundColor: 'rgba(52,152,219,0.7)'
          }]
        },
        options: {
          responsive: true,
          animation: false,
          scales: {
            x: { ticks: { color: '#aaa' }, grid: { display: false } },
            y: {
              ticks: { color: '#aaa', callback: (v) => v + ' ' + yLabel },
              grid: { color: 'rgba(255,255,255,0.05)' }
            }
          },
          plugins: { legend: { labels: { color: '#eee' } } }
        }
      });
    }

    function makeHeatmapChart(ctx) {
      return new Chart(ctx, {
        type: 'matrix',
        data: {
          datasets: [{
            label: 'ΔMesh-Time Heatmap',
            data: [],
            borderWidth: 0,
            backgroundColor: (context) => {
              const chart = context.chart;
              const raw = context.raw;
              if (!raw || typeof raw.v !== 'number') return 'rgba(0,0,0,0)';
              const value = raw.v;
              const maxV = chart._maxV || 1;
              const ratio = Math.min(1, value / maxV);
              const r = Math.round(255 * ratio);
              const g = Math.round(255 * (1 - ratio));
              return `rgba(${r},${g},150,0.8)`;
            },
            width: (context) => {
              const chart = context.chart;
              const area = chart.chartArea || {};
              const nBins = chart._nBins || 1;
              const w = (area.right - area.left) / nBins;
              return (Number.isFinite(w) && w > 0) ? w : 10;
            },
            height: (context) => {
              const chart = context.chart;
              const area = chart.chartArea || {};
              const cats = chart._categories || ['pair'];
              const nCats = cats.length || 1;
              const h = (area.bottom - area.top) / nCats;
              return (Number.isFinite(h) && h > 0) ? h : 10;
            }
          }]
        },
        options: {
          responsive: true,
          animation: false,
          scales: {
            x: {
              type: 'time',
              time: { unit: 'second' },
              position: 'bottom',
              ticks: { color: '#aaa' },
              grid: { color: 'rgba(255,255,255,0.05)' }
            },
            y: {
              type: 'linear',
              ticks: {
                color: '#aaa',
                callback: function (value) {
                  const chart = this.chart;
                  const cats = chart._categories || [];
                  const idx = Math.round(value);
                  return cats[idx] || '';
                }
              },
              grid: { color: 'rgba(255,255,255,0.05)' }
            }
          },
          plugins: {
            legend: { labels: { color: '#eee' } },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const chart = ctx.chart;
                  const cats = chart._categories || [];
                  const idx = Math.round(ctx.raw.y);
                  const pairId = cats[idx] || '?';
                  const v = (ctx.raw && typeof ctx.raw.v === 'number') ? ctx.raw.v : 0;
                  const t = new Date(ctx.raw.x);
                  return `${pairId} @ ${t.toLocaleTimeString()} : ${v.toFixed(2)} ms`;
                }
              }
            }
          }
        }
      });
    }

    function initCharts() {
      pairChart        = makeLineChart(document.getElementById('pairChart').getContext('2d'), 'ms');
      offsetDeltaChart = makeLineChart(document.getElementById('offsetDeltaChart').getContext('2d'), 'ms');
      jitterBarChart   = makeBarChart(document.getElementById('jitterBarChart').getContext('2d'), 'ms');
      heatmapChart     = makeHeatmapChart(document.getElementById('heatmapChart').getContext('2d'));
    }

    function updatePairChart(chart, pairs) {
      const pairIds = Object.keys(pairs).sort();
      chart.data.datasets = [];
      pairIds.forEach((pairId, idx) => {
        const baseColor   = colors[idx % colors.length];
        const strokeColor = baseColor.replace('0.5', '0.9');
        const points = pairs[pairId].map(p => ({ x: new Date(p.t_wall * 1000), y: p.delta_ms }));
        chart.data.datasets.push({
          label: pairId,
          data: points,
          borderColor: strokeColor,
          backgroundColor: baseColor,
          fill: false,
          pointRadius: 0,
          borderWidth: 1.5
        });
      });
      chart.update();
    }

    function updateOffsetDeltaChart(chart, series) {
      const nodeIds = Object.keys(series).sort();
      chart.data.datasets = [];
      nodeIds.forEach((nodeId, idx) => {
        const baseColor   = colors[idx % colors.length];
        const strokeColor = baseColor.replace('0.5', '0.9');
        const points = series[nodeId].map(p => ({ x: new Date(p.t_wall * 1000), y: p.delta_offset_ms }));
        chart.data.datasets.push({
          label: `Node ${nodeId}`,
          data: points,
          borderColor: strokeColor,
          backgroundColor: baseColor,
          fill: false,
          pointRadius: 0,
          borderWidth: 1.5
        });
      });
      chart.update();
    }

    function updateJitterBarChart(chart, jitterPairs) {
      const ids = Object.keys(jitterPairs || {}).sort();
      const labels = [];
      const data = [];
      ids.forEach(id => {
        const v = jitterPairs[id];
        if (v !== null && v !== undefined && Number.isFinite(v)) {
          labels.push(id);
          data.push(v);
        }
      });
      chart.data.labels = labels;
      chart.data.datasets[0].data = data;
      chart.update();
    }

    function updateHeatmapChart(chart, heatmap) {
      const data = (heatmap && heatmap.data) || [];
      if (!data.length) {
        chart.data.datasets[0].data = [];
        chart._categories = [];
        chart._nBins = 1;
        chart._maxV = 1;
        chart.update();
        return;
      }

      const categories = Array.from(new Set(data.map(d => d.pair))).sort();
      const catIndex = new Map(categories.map((c, i) => [c, i]));

      const nBins = heatmap.n_bins || 1;
      let maxV = 0;

      const matrixData = data.map(d => {
        const v = Math.abs(d.value);
        if (v > maxV) maxV = v;
        return { x: new Date(d.t_bin * 1000), y: catIndex.get(d.pair) ?? 0, v: v };
      });

      chart.data.datasets[0].data = matrixData;
      chart._categories = categories;
      chart._nBins = nBins;
      chart._maxV = maxV || 1;
      chart.update();
    }

    async function fetchNtpData() {
      const resp = await fetch('/api/ntp_timeseries');
      return await resp.json();
    }

    async function fetchTopology() {
      const resp = await fetch('/api/topology');
      return await resp.json();
    }

    function drawMesh(canvas, topo, activeNodes) {
      const ctx = canvas.getContext('2d');
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      if (!topo.nodes || topo.nodes.length === 0) {
        ctx.fillStyle = '#777';
        ctx.font = '12px system-ui';
        ctx.fillText('Keine Nodes in config/nodes.json gefunden.', 10, 20);
        return;
      }

      const n = topo.nodes.length;
      const radius = Math.min(w, h) * 0.35;
      const cx = w / 2;
      const cy = h / 2;

      const positions = {};
      topo.nodes.forEach((node, idx) => {
        const angle = (2 * Math.PI * idx) / n - Math.PI / 2;
        positions[node.id] = { x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle) };
      });

      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1.0;
      (topo.links || []).forEach(link => {
        const a = positions[link.source];
        const b = positions[link.target];
        if (!a || !b) return;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      });

      (topo.nodes || []).forEach(node => {
        const pos = positions[node.id];
        if (!pos) return;
        const active = activeNodes.has(node.id);
        const isRoot = !!node.is_root;
        const r = isRoot ? 20 : 16;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = active ? 'rgba(39,174,96,0.25)' : 'rgba(149,165,166,0.2)';
        ctx.fill();

        ctx.lineWidth = isRoot ? 3.0 : (active ? 2.0 : 1.0);
        ctx.strokeStyle = isRoot ? '#f1c40f' : (active ? '#2ecc71' : '#7f8c8d');
        ctx.stroke();

        ctx.fillStyle = '#ecf0f1';
        ctx.font = '12px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.id, pos.x, pos.y);
      });

      const roots = (topo.nodes || []).filter(n => n.is_root);
      ctx.fillStyle = '#aaa';
      ctx.font = '11px system-ui';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      const rootLabel = roots.length ? ('Root: ' + roots.map(r => r.id).join(', ')) : 'Root: keiner (symmetrisches Mesh)';
      ctx.fillText(rootLabel, 10, 10);
    }

    async function refresh() {
      try {
        const [ntpData, topo] = await Promise.all([fetchNtpData(), fetchTopology()]);
        const series  = ntpData.series || {};
        const pairs   = ntpData.pairs || {};
        const jitter  = ntpData.jitter_sigma_pairs || {}; // robust IQR-sigma
        const heatmap = ntpData.heatmap || {data: [], n_bins: 0};

        updatePairChart(pairChart, pairs);
        updateOffsetDeltaChart(offsetDeltaChart, series);
        updateJitterBarChart(jitterBarChart, jitter);
        updateHeatmapChart(heatmapChart, heatmap);

        const activeNodes = new Set(Object.keys(series || {}));
        const meshCanvas  = document.getElementById('meshCanvas');
        meshCanvas.width  = meshCanvas.clientWidth;
        meshCanvas.height = meshCanvas.clientHeight;
        drawMesh(meshCanvas, topo, activeNodes);
      } catch (e) {
        console.error('refresh failed', e);
      }
    }

    window.addEventListener('load', () => {
      initCharts();
      refresh();
      setInterval(refresh, 2000);
    });
  </script>
</body>
</html>
"""


def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Storage(str(DB_PATH))  # create schema if missing


if __name__ == "__main__":
    ensure_db()
    print("Starting MeshTime Web-UI on http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)
