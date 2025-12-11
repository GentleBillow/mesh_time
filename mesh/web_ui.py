# web_ui.py
# MeshTime Web-UI mit 5 Plots + Mesh-Topologie + Live-Updates

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template_string, jsonify

DB_PATH = Path("mesh_data.sqlite")
CFG_PATH = Path("config/nodes.json")

LED_PERIOD_S = 0.5  # 500 ms

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
      - nodes: [{id, ip, color}]
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
        nodes.append({"id": node_id, "ip": ip, "color": color})

        for neigh in entry.get("neighbors", []):
            # doppelte Kanten vermeiden
            if node_id < neigh:
                links.append({"source": node_id, "target": neigh})

    return {"nodes": nodes, "links": links}


def get_ntp_timeseries(window_seconds=600, max_points=1000, root_id="C"):
    """
    Holt die letzten NTP-Referenzen aus der DB und baut:
      - pro Node eine Zeitreihe von:
        t_wall, err_ms, offset_ms, phase_ms, delta_vs_root_ms
      - jitter_ms pro Node (stddev der err_ms)
    """
    conn = get_conn()
    cur = conn.cursor()

    # Hole die letzten max_points aus dem gewünschten Zeitfenster
    rows = cur.execute(
        """
        SELECT node_id, t_wall, t_mesh, offset, err_mesh_vs_wall, created_at
        FROM ntp_reference
        WHERE created_at >= strftime('%s','now', ?)
        ORDER BY id ASC
        LIMIT ?
        """,
        (f"-{int(window_seconds)} seconds", max_points),
    ).fetchall()
    conn.close()

    # in Python strukturieren
    per_node = {}
    for r in rows:
        node = r["node_id"]
        per_node.setdefault(node, []).append(r)

    # Letzte Root-Fehler (approx) pro Zeitpunkt: wir nehmen einfach
    # den zuletzt gesehenen Fehler der Root, wenn kein Zeitabgleich möglich.
    # Für eine Lab-Visualisierung reicht das aus.
    def build_series():
        series = {}
        err_by_node = {}  # für Jitter
        root_last_err = None

        # wir gehen wieder durch alle Rows und bauen für jeden Node Punkte
        for node, node_rows in per_node.items():
            if node not in series:
                series[node] = []
                err_by_node[node] = []

        # Wir iterieren lieber in globaler Zeitreihenfolge
        all_rows_sorted = sorted(rows, key=lambda r: r["t_wall"])

        for r in all_rows_sorted:
            node = r["node_id"]
            t_wall = float(r["t_wall"])
            t_mesh = float(r["t_mesh"])
            offset = float(r["offset"])
            err = float(r["err_mesh_vs_wall"])

            err_ms = err * 1000.0
            offset_ms = offset * 1000.0

            # LED-Phasenfehler relativ zu Flanke:
            # Periodensignal P, Phase in [0, P),
            # Symmetrische Distanz zur nächsten idealen Flanke (0 oder P/2)
            phi = t_mesh % LED_PERIOD_S
            phase_err_ms = min(phi, LED_PERIOD_S - phi) * 1000.0

            # Root-Fehler tracken
            if node == root_id:
                root_last_err = err_ms

            # ΔMesh-Time gegen Root als Differenz der Fehler
            if root_last_err is not None:
                delta_vs_root_ms = err_ms - root_last_err
            else:
                delta_vs_root_ms = None

            point = {
                "t_wall": t_wall,
                "err_ms": err_ms,
                "offset_ms": offset_ms,
                "phase_ms": phase_err_ms,
                "delta_vs_root_ms": delta_vs_root_ms,
            }
            series[node].append(point)
            err_by_node[node].append(err_ms)

        # Jitter (stddev) pro Node
        jitter_ms = {}
        for node, errs in err_by_node.items():
            if len(errs) < 2:
                jitter_ms[node] = None
                continue
            mean = sum(errs) / len(errs)
            var = sum((e - mean) ** 2 for e in errs) / (len(errs) - 1)
            jitter_ms[node] = var ** 0.5

        return series, jitter_ms

    series, jitter_ms = build_series()
    return {
        "root_id": root_id,
        "series": series,
        "jitter_ms": jitter_ms,
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
    data = get_ntp_timeseries(window_seconds=600, max_points=1000, root_id="C")
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
  <!-- Chart.js + Date-Adapter -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
</head>
<body>
  <h1>MeshTime Monitor</h1>
  <p class="small">
    Datenquelle: <code>{{ db_path }}</code> (ntp_reference)
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
        <p class="small">Knoten und Links basierend auf <code>config/nodes.json</code>. Grün = Node hat NTP-Daten, grau = bisher keine.</p>
      </div>
    </div>

    <!-- rechte Seite: Plots -->
    <div class="card">
      <h2>Mesh-Time Plots (letzte 10 Minuten)</h2>
      <div class="plots-grid">
        <div>
          <h3 style="font-size:0.95rem;">1) Mesh − Wallclock Fehler</h3>
          <canvas id="errChart" height="160"></canvas>
        </div>
        <div>
          <h3 style="font-size:0.95rem;">2) Offset pro Node</h3>
          <canvas id="offsetChart" height="160"></canvas>
        </div>
        <div>
          <h3 style="font-size:0.95rem;">3) ΔMesh-Time vs. Root</h3>
          <canvas id="deltaChart" height="160"></canvas>
        </div>
        <div>
          <h3 style="font-size:0.95rem;">4) LED-Phasenfehler</h3>
          <canvas id="phaseChart" height="160"></canvas>
        </div>
        <div>
          <h3 style="font-size:0.95rem;">5) Jitter (StdDev Fehler)</h3>
          <canvas id="jitterChart" height="160"></canvas>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    MeshTime Web-UI – Wenn alle Kurven flach &lt;5 ms sind und das Mesh grün leuchtet, hast du einen kleinen Zeit-Konsens-Computer gebaut.
  </div>

  <script>
    const colors = [
      'rgba(46, 204, 113, 1)',
      'rgba(52, 152, 219, 1)',
      'rgba(231, 76, 60, 1)',
      'rgba(241, 196, 15, 1)',
      'rgba(155, 89, 182, 1)'
    ];

    let errChart, offsetChart, deltaChart, phaseChart, jitterChart;

    function makeLineChart(ctx, labelSuffix, yLabel) {
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
                label: (ctx) => `${ctx.dataset.label}${labelSuffix}: ${ctx.parsed.y.toFixed(2)} ${yLabel}`
              }
            }
          }
        }
      });
    }

    function makeBarChart(ctx, yLabel) {
      return new Chart(ctx, {
        type: 'bar',
        data: { labels: [], datasets: [{
          label: 'Jitter',
          data: [],
          backgroundColor: 'rgba(52, 152, 219, 0.7)'
        }] },
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
            }
          }
        }
      });
    }

    function initCharts() {
      errChart    = makeLineChart(document.getElementById('errChart').getContext('2d'), '', 'ms');
      offsetChart = makeLineChart(document.getElementById('offsetChart').getContext('2d'), '', 'ms');
      deltaChart  = makeLineChart(document.getElementById('deltaChart').getContext('2d'), ' vs Root', 'ms');
      phaseChart  = makeLineChart(document.getElementById('phaseChart').getContext('2d'), '', 'ms');
      jitterChart = makeBarChart(document.getElementById('jitterChart').getContext('2d'), 'ms');
    }

    function updateLineChart(chart, series, field, rootId, skipRootInDelta=false) {
      const nodeIds = Object.keys(series).sort();
      chart.data.datasets = [];
      nodeIds.forEach((nodeId, idx) => {
        if (skipRootInDelta && nodeId === rootId) return;
        const color = colors[idx % colors.length];
        const points = series[nodeId]
          .filter(p => p[field] !== null && p[field] !== undefined)
          .map(p => ({ x: new Date(p.t_wall * 1000), y: p[field] }));
        chart.data.datasets.push({
          label: `Node ${nodeId}`,
          data: points,
          borderColor: color,
          backgroundColor: color,
          fill: false,
          tension: 0.1,
          pointRadius: 0
        });
      });
      chart.update();
    }

    function updateJitterChart(chart, jitter_ms) {
      const nodeIds = Object.keys(jitter_ms).sort();
      chart.data.labels = nodeIds;
      chart.data.datasets[0].data = nodeIds.map(node => jitter_ms[node] || 0);
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
        const r = 16;

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r, 0, 2 * Math.PI);
        ctx.fillStyle = active ? 'rgba(39, 174, 96, 0.25)' : 'rgba(149, 165, 166, 0.2)';
        ctx.fill();

        ctx.lineWidth = active ? 2.0 : 1.0;
        ctx.strokeStyle = active ? '#2ecc71' : '#7f8c8d';
        ctx.stroke();

        ctx.fillStyle = '#ecf0f1';
        ctx.font = '12px system-ui';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(node.id, pos.x, pos.y);
      });
    }

    async function refresh() {
      try {
        const [ntpData, topo] = await Promise.all([
          fetchNtpData(),
          fetchTopology()
        ]);

        const series = ntpData.series || {};
        const rootId = ntpData.root_id;

        updateLineChart(errChart,    series, 'err_ms',          rootId);
        updateLineChart(offsetChart, series, 'offset_ms',       rootId);
        updateLineChart(deltaChart,  series, 'delta_vs_root_ms', rootId, true);
        updateLineChart(phaseChart,  series, 'phase_ms',        rootId);
        updateJitterChart(jitterChart, ntpData.jitter_ms || {});

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
