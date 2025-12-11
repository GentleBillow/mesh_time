# web_ui.py
# Minimaler Flask-Webserver, der die mesh_data.sqlite visualisiert.

from flask import Flask, render_template_string
import sqlite3
from datetime import datetime

DB_PATH = "mesh_data.sqlite"

app = Flask(__name__)


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.route("/")
def index():
    conn = get_conn()
    cur = conn.cursor()

    # Letzter NTP-Referenzpunkt pro Node
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

    # Zeitreihe der letzten 200 Punkte (für Plot)
    series = cur.execute(
        """
        SELECT node_id, t_wall, t_mesh, offset, err_mesh_vs_wall, created_at
        FROM ntp_reference
        ORDER BY id DESC
        LIMIT 200
        """
    ).fetchall()
    conn.close()

    # Für Chart.js vorbereiten
    timeline = [
        {
            "node_id": row["node_id"],
            "t_wall": row["t_wall"],
            "t_mesh": row["t_mesh"],
            "offset_ms": row["offset"] * 1000.0,
            "err_ms": row["err_mesh_vs_wall"] * 1000.0,
            "created_at": row["created_at"],
        }
        for row in series
    ]

    return render_template_string(TEMPLATE, last_by_node=last_by_node, timeline=timeline)


TEMPLATE = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>MeshTime Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Ganz simples Styling -->
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      padding: 1.5rem;
      background: #111;
      color: #eee;
    }
    h1, h2 {
      margin-top: 0;
    }
    .grid {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, 1.5fr);
      gap: 1.5rem;
      align-items: flex-start;
    }
    @media (max-width: 900px) {
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
    .footer {
      margin-top: 1rem;
      font-size: 0.8rem;
      opacity: 0.6;
    }
  </style>
  <!-- Chart.js von CDN -->
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
</head>
<body>
  <h1>MeshTime Monitor</h1>
  <p class="small">
    Datenquelle: <code>{{ DB_PATH if DB_PATH is defined else "mesh_data.sqlite" }}</code> (ntp_reference)
  </p>

  <div class="grid">
    <!-- Linke Karte: aktueller Status pro Node -->
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

    <!-- Rechte Karte: Zeitreihe -->
    <div class="card">
      <h2>Mesh vs. Wallclock</h2>
      <canvas id="errChart" height="200"></canvas>
      <p class="small">
        Gezeigt werden die letzten {{ timeline|length }} Punkte aus <code>ntp_reference</code>.
      </p>
    </div>
  </div>

  <div class="card" style="margin-top:1.5rem;">
    <h2>Raw-Stats</h2>
    <pre class="small" id="rawStats"></pre>
  </div>

  <div class="footer">
    MeshTime Web-UI – nur für Laborzwecke. Wenn die Kurve flach bei &lt;5 ms ist, hast du gewonnen.
  </div>

  <script>
    // Daten aus Flask
    const timeline = {{ timeline | tojson }};

    // In JS-Struktur für Chart.js umformen (Gruppierung nach Node)
    const byNode = {};
    for (const row of timeline) {
      const node = row.node_id;
      if (!byNode[node]) byNode[node] = [];
      // t_wall ist Unix-Zeit in Sekunden
      const t = new Date(row.t_wall * 1000);
      byNode[node].push({
        x: t,
        y: row.err_ms
      });
    }

    const colors = [
      'rgba(46, 204, 113, 1)',
      'rgba(52, 152, 219, 1)',
      'rgba(231, 76, 60, 1)',
      'rgba(241, 196, 15, 1)'
    ];

    const datasets = Object.keys(byNode).sort().map((node, idx) => ({
      label: `Node ${node}`,
      data: byNode[node],
      borderColor: colors[idx % colors.length],
      backgroundColor: colors[idx % colors.length],
      fill: false,
      tension: 0.1,
      pointRadius: 0
    }));

    const ctx = document.getElementById('errChart').getContext('2d');
    const errChart = new Chart(ctx, {
      type: 'line',
      data: { datasets },
      options: {
        responsive: true,
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'second'
            },
            ticks: { color: '#aaa' },
            grid: { color: 'rgba(255,255,255,0.05)' }
          },
          y: {
            ticks: {
              color: '#aaa',
              callback: (v) => v + ' ms'
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
              label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} ms`
            }
          }
        }
      }
    });

    // Raw-Stats anzeigen
    const stats = {};
    for (const row of timeline) {
      const node = row.node_id;
      if (!stats[node]) stats[node] = { count: 0, min: +Infinity, max: -Infinity };
      stats[node].count += 1;
      stats[node].min = Math.min(stats[node].min, row.err_ms);
      stats[node].max = Math.max(stats[node].max, row.err_ms);
    }
    document.getElementById('rawStats').textContent = JSON.stringify(stats, null, 2);
  </script>
</body>
</html>
"""

# kleiner Filter für UTC-Ausgabe im Template
@app.template_filter("datetime_utc")
def datetime_utc(ts):
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


if __name__ == "__main__":
    print("Starting MeshTime Web-UI on http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)
