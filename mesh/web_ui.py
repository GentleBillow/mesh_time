# web_ui.py
# MeshTime Web-UI mit relativen Plots (paarweise ΔMesh-Time) + Topologie

import json
import math
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template_string, jsonify

DB_PATH = Path("mesh_data.sqlite")
CFG_PATH = Path("config/nodes.json")

app = Flask(__name__)


# ------------------------------------------------------------
# Hilfsfunktionen Backend
# ------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def load_config():
    if not CFG_PATH.exists():
        return {}
    with CFG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_topology():
    """
    Liest config/nodes.json und baut eine einfache Mesh-Topologie:
      - nodes: [{id, ip, color, is_root}]
      - links: [{source, target}]
      - has_root: bool
    """
    cfg = load_config()
    nodes = []
    links = []
    has_root = False

    for node_id, entry in cfg.items():
        if node_id == "sync":
            continue

        ip = entry.get("ip")
        color = entry.get("color") or "#3498db"
        sync_cfg = entry.get("sync", {}) or {}
        is_root = bool(sync_cfg.get("is_root", False))
        if is_root:
            has_root = True

        nodes.append(
            {
                "id": node_id,
                "ip": ip,
                "color": color,
                "is_root": is_root,
            }
        )

        for neigh in entry.get("neighbors", []):
            # doppelte Kanten vermeiden
            if node_id < neigh:
                links.append({"source": node_id, "target": neigh})

    return {"nodes": nodes, "links": links, "has_root": has_root}


def _quantiles(values):
    """
    Einfache Quantils-Berechnung (Q1, Median, Q3) mit linearer Interpolation.
    Liefert: (min, q1, median, q3, max)
    """
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return None

    def q(p: float) -> float:
        if n == 1:
            return xs[0]
        pos = p * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        frac = pos - lo
        return xs[lo] * (1.0 - frac) + xs[hi] * frac

    return xs[0], q(0.25), q(0.5), q(0.75), xs[-1]


def get_ntp_timeseries(window_seconds=600, max_points=2000):
    """
    Holt die letzten ntp_reference-Samples und baut:
      - series_per_node: pro Node Zeitreihe mit
            t_wall, offset_ms, delta_offset_ms
      - pairs: paarweise ΔMesh-Time (Node i vs. j) als Zeitreihe
            t_wall, delta_ms
      - jitter_pairs: Boxplot-Stats der ΔMesh-Time pro Paar
            {pair_id: {min,q1,median,q3,max}}
    ΔMesh-Time wird aus den Fehlern err_mesh_vs_wall berechnet, d.h.
    Δ_ij ≈ (mesh_i - wall) - (mesh_j - wall) = err_i - err_j.
    Wallclock fällt also raus, wir nutzen es nur als gemeinsame Referenz.
    """
    conn = get_conn()
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT node_id, t_wall, t_mesh, offset, err_mesh_vs_wall, created_at
        FROM ntp_reference
        ORDER BY id ASC
        """
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "series": {},
            "pairs": {},
            "jitter_pairs": {},
        }

    # wir beschränken das Fenster in Python (letzte window_seconds)
    newest_ts = max(float(r["created_at"]) for r in rows)
    cutoff = newest_ts - float(window_seconds)
    rows = [r for r in rows if float(r["created_at"]) >= cutoff]

    # nach Node gruppieren
    per_node = {}
    for r in rows:
        per_node.setdefault(r["node_id"], []).append(r)

    # Node-Zeitreihen mit Offset + Delta-Offset
    series = {}
    for node_id, node_rows in per_node.items():
        node_rows_sorted = sorted(node_rows, key=lambda r: r["t_wall"])
        node_series = []
        prev_offset_ms = None
        for r in node_rows_sorted:
            t_wall = float(r["t_wall"])
            offset_ms = float(r["offset"]) * 1000.0
            if prev_offset_ms is None:
                delta_offset_ms = 0.0
            else:
                delta_offset_ms = offset_ms - prev_offset_ms
            prev_offset_ms = offset_ms

            node_series.append(
                {
                    "t_wall": t_wall,
                    "offset_ms": offset_ms,
                    "delta_offset_ms": delta_offset_ms,
                }
            )
        series[node_id] = node_series

    # Paarweise ΔMesh-Time (err_i - err_j)
    all_rows_sorted = sorted(rows, key=lambda r: r["t_wall"])
    last_err_ms = {}  # node_id -> letzter Fehler in ms
    pair_series = {}  # "A-B" -> [{"t_wall":..,"delta_ms":..}, ...]

    def norm_pair(a, b):
        return "-".join(sorted([a, b]))

    for r in all_rows_sorted:
        node = r["node_id"]
        t_wall = float(r["t_wall"])
        err_ms = float(r["err_mesh_vs_wall"]) * 1000.0
        last_err_ms[node] = err_ms

        # sobald wir mindestens zwei Nodes gesehen haben, können wir
        # für alle bekannten Paare Δ berechnen
        nodes_now = sorted(last_err_ms.keys())
        if len(nodes_now) < 2:
            continue

        for i in range(len(nodes_now)):
            for j in range(i + 1, len(nodes_now)):
                a = nodes_now[i]
                b = nodes_now[j]
                pair_id = norm_pair(a, b)
                delta_ms = last_err_ms[a] - last_err_ms[b]
                pair_series.setdefault(pair_id, []).append(
                    {
                        "t_wall": t_wall,
                        "delta_ms": delta_ms,
                    }
                )

    # Jitter-Boxplots über ΔMesh-Time pro Paar
    jitter_pairs = {}
    for pair_id, pts in pair_series.items():
        if len(pts) < 3:
            continue
        deltas = [p["delta_ms"] for p in pts]
        stats = _quantiles(deltas)
        if stats is None:
            continue
        min_v, q1, med, q3, max_v = stats
        jitter_pairs[pair_id] = {
            "min": min_v,
            "q1": q1,
            "median": med,
            "q3": q3,
            "max": max_v,
        }

    return {
        "series": series,
        "pairs": pair_series,
        "jitter_pairs": jitter_pairs,
    }


# ------------------------------------------------------------
# Flask Routen
# ------------------------------------------------------------

@app.route("/")
def index():
    # für die Kopf-Tabelle: letzter Eintrag pro Node
    conn = get_conn()
    cur = conn.cursor()
    last_by_node = cur.execute(
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
    data = get_ntp_timeseries(window_seconds=600, max_points=2000)
    return jsonify(data)


# Jinja Filter
@app.template_filter("datetime_utc")
def datetime_utc(ts):
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


# ------------------------------------------------------------
# HTML + JS Template
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
    h1, h2, h3 {
      margin-top: 0;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, 2fr);
      gap: 1.5rem;
      align-items: flex-start;
    }
    @media (max-width: 1100px) {
      .grid {
        grid-template-columns: minmax(0, 1fr);
      }
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
    }
    th {
      border-bottom: 1px solid rgba(255,255,255,0.15);
      font-weight: 600;
    }
    tr:nth-child(even) td {
      background: rgba(255,255,255,0.02);
    }
    .pill {
      display: inline-block;
      padding: 0.1rem 0.5rem;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 500;
    }
    .pill-ok {
      background: rgba(46, 204, 113, 0.12);
      color: #2ecc71;
    }
    .pill-warn {
      background: rgba(241, 196, 15, 0.12);
      color: #f1c40f;
    }
    .pill-bad {
      background: rgba(231, 76, 60, 0.12);
      color: #e74c3c;
    }
    .small {
      font-size: 0.8rem;
      opacity: 0.7;
    }
    canvas {
      max-width: 100%;
    }
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
      .plots-grid {
        grid-template-columns: minmax(0, 1fr);
      }
    }
    .footer {
      margin-top: 1rem;
      font-size: 0.8rem;
      opacity: 0.6;
    }
  </style>
  <!-- Chart.js + Date-Adapter + BoxPlot-Plugin -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-box-and-violin-plot@4.3.0/build/Chart.BoxPlot.js"></script>
</head>
<body>
  <h1>MeshTime Monitor</h1>
  <p class="small">
    Datenquelle: <code>{{ db_path }}</code> (ntp_reference, relativ ausgewertet)
  </p>

  <div class="grid">
    <!-- linke Seite: Tabelle + Mesh-Topologie -->
    <div>
      <div class="card">
        <h2>Aktueller Status pro Node</h2>
        {% if last_by_node %}
        <table>
          <thead>
            <tr>
              <th>Node</th>
              <th>Offset</th>
              <th>Mesh − Wallclock</th>
              <th>Wallclock (UTC)</th>
            </tr>
          </thead>
          <tbody>
            {% for row in last_by_node %}
            {% set err_ms = row["err_mesh_vs_wall"] * 1000.0 %}
            {% set offset_ms = row["offset"] * 1000.0 %}
            {% if err_ms|abs < 5 %}
              {% set cls = "pill-ok" %}
            {% elif err_ms|abs < 20 %}
              {% set cls = "pill-warn" %}
            {% else %}
              {% set cls = "pill-bad" %}
            {% endif %}
            <tr>
              <td>{{ row["node_id"] }}</td>
              <td><span class="pill {{ cls }}">{{ "%.2f" % offset_ms }} ms</span></td>
              <td>{{ "%.2f" % err_ms }} ms</td>
              <td class="small">
                {{ row["t_wall"] | float | int | datetime_utc }}
              </td>
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
          Knoten und Links basierend auf <code>config/nodes.json</code>.
          Grün = Node hat Daten, grau = bisher keine.
          Dicker Rand = <strong>Root</strong> (falls vorhanden).
        </p>
      </div>
    </div>

    <!-- rechte Seite: Plots -->
    <div class="card">
      <h2>Mesh-Time Plots (letzte 10 Minuten)</h2>
      <div class="plots-grid">
        <div>
          <h3 style="font-size:0.95rem;">1) Paarweise ΔMesh-Time</h3>
          <canvas id="pairChart" height="160"></canvas>
        </div>
        <div>
          <h3 style="font-size:0.95rem;">2) Offset-Änderung pro Node (Δoffset)</h3>
          <canvas id="offsetDeltaChart" height="160"></canvas>
        </div>
        <div>
          <h3 style="font-size:0.95rem;">3) Jitter der ΔMesh-Time (Boxplots pro Paar)</h3>
          <canvas id="jitterBoxChart" height="160"></canvas>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    MeshTime Web-UI – Wenn alle ΔMesh-Linien flach &lt;5 ms liegen, hast du einen kleinen Zeit-Konsens-Computer gebaut.
  </div>

  <script>
    const colors = [
      'rgba(46, 204, 113, 0.9)',
      'rgba(52, 152, 219, 0.9)',
      'rgba(231, 76, 60, 0.9)',
      'rgba(241, 196, 15, 0.9)',
      'rgba(155, 89, 182, 0.9)'
    ];

    let pairChart, offsetDeltaChart, jitterBoxChart;

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
                callback: (v) => v.toFixed ? v.toFixed(1) + ' ' + yLabel : v + ' ' + yLabel
              },
              grid: { color: 'rgba(255,255,255,0.05)' }
            }
          },
          plugins: {
            legend: {
              labels: { color: '#eee' }
            },
            tooltip: {
              callbacks: {
                label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} ${yLabel}`
              }
            }
          },
          elements: {
            line: {
              tension: 0.1
            },
            point: {
              radius: 0
            }
          }
        }
      });
    }

    function makeBoxPlotChart(ctx, yLabel) {
      return new Chart(ctx, {
        type: 'boxplot',
        data: {
          labels: [],
          datasets: [{
            label: 'ΔMesh-Time',
            data: [],
            backgroundColor: 'rgba(52, 152, 219, 0.6)',
            borderColor: 'rgba(52, 152, 219, 1)',
          }]
        },
        options: {
          responsive: true,
          animation: false,
          scales: {
            x: {
              ticks: { color: '#aaa' },
              grid: { display: false }
            },
            y: {
              ticks: {
                color: '#aaa',
                callback: (v) => v + ' ' + yLabel
              },
              grid: { color: 'rgba(255,255,255,0.05)' }
            }
          },
          plugins: {
            legend: {
              labels: { color: '#eee' }
            },
            tooltip: {
              callbacks: {
                label: (ctx) => {
                  const v = ctx.raw;
                  return [
                    `min: ${v.min.toFixed(2)} ${yLabel}`,
                    `Q1 : ${v.q1.toFixed(2)} ${yLabel}`,
                    `med: ${v.median.toFixed(2)} ${yLabel}`,
                    `Q3 : ${v.q3.toFixed(2)} ${yLabel}`,
                    `max: ${v.max.toFixed(2)} ${yLabel}`,
                  ];
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
      jitterBoxChart   = makeBoxPlotChart(document.getElementById('jitterBoxChart').getContext('2d'), 'ms');
    }

    function updatePairChart(chart, pairs) {
      const pairIds = Object.keys(pairs).sort();
      chart.data.datasets = [];
      pairIds.forEach((pairId, idx) => {
        const color = colors[idx % colors.length];
        const points = pairs[pairId].map(p => ({
          x: new Date(p.t_wall * 1000),
          y: p.delta_ms
        }));
        chart.data.datasets.push({
          label: pairId,
          data: points,
          borderColor: color,
          backgroundColor: color,
          fill: false,
          pointRadius: 0,
          borderWidth: 1.5,
          // leichte Transparenz für Überlagerung
          borderColor: color.replace('0.9', '0.5'),
          backgroundColor: color.replace('0.9', '0.2'),
        });
      });
      chart.update();
    }

    function updateOffsetDeltaChart(chart, series) {
      const nodeIds = Object.keys(series).sort();
      chart.data.datasets = [];
      nodeIds.forEach((nodeId, idx) => {
        const color = colors[idx % colors.length];
        const points = series[nodeId].map(p => ({
          x: new Date(p.t_wall * 1000),
          y: p.delta_offset_ms
        }));
        chart.data.datasets.push({
          label: `Node ${nodeId}`,
          data: points,
          borderColor: color,
          backgroundColor: color,
          fill: false,
          pointRadius: 0,
          borderWidth: 1.5
        });
      });
      chart.update();
    }

    function updateJitterBoxChart(chart, jitter_pairs) {
      const pairIds = Object.keys(jitter_pairs).sort();
      chart.data.labels = pairIds;
      chart.data.datasets[0].data = pairIds.map(pairId => jitter_pairs[pairId]);
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
        const x = cx + radius * Math.cos(angle);
        const y = cy + radius * Math.sin(angle);
        positions[node.id] = { x, y };
      });

      // Links
      ctx.strokeStyle = 'rgba(255,255,255,0.25)';
      ctx.lineWidth = 1.0;
      topo.links.forEach(link => {
        const a = positions[link.source];
        const b = positions[link.target];
        if (!a || !b) return;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      });

      // Nodes
      topo.nodes.forEach(node => {
        const pos = positions[node.id];
        if (!pos) return;
        const active = activeNodes.has(node.id);
        const isRoot = !!node.is_root;
        const r = isRoot ? 20 : 16;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = active ? 'rgba(39, 174, 96, 0.25)' : 'rgba(149, 165, 166, 0.2)';
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

      // Root-Info
      const roots = topo.nodes.filter(n => n.is_root);
      ctx.fillStyle = '#aaa';
      ctx.font = '11px system-ui';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      const rootLabel = roots.length
        ? 'Root: ' + roots.map(r => r.id).join(', ')
        : 'Root: keiner (voll symmetrisch)';
      ctx.fillText(rootLabel, 10, 10);
    }

    async function refresh() {
      try {
        const [ntpData, topo] = await Promise.all([
          fetchNtpData(),
          fetchTopology()
        ]);

        const series = ntpData.series || {};
        const pairs = ntpData.pairs || {};
        const jitter_pairs = ntpData.jitter_pairs || {};

        updatePairChart(pairChart, pairs);
        updateOffsetDeltaChart(offsetDeltaChart, series);
        updateJitterBoxChart(jitterBoxChart, jitter_pairs);

        // aktive Nodes = alle, die Daten haben
        const activeNodes = new Set(Object.keys(series || {}));
        const meshCanvas = document.getElementById('meshCanvas');
        // Canvas-Size anpassen
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
      setInterval(refresh, 2000); // alle 2s live updaten
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Starting MeshTime Web-UI on http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)
