# -*- coding: utf-8 -*-
# mesh/web_ui.py — CLEAN REWRITE
# Python 3.7 compatible

import math
import json
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from flask import Flask, jsonify, render_template_string

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "mesh_data.sqlite"
CFG_PATH = BASE_DIR / "config" / "nodes.json"

from mesh.storage import Storage

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = Flask(__name__)

WINDOW_S_DEFAULT = 600.0
BIN_S_DEFAULT = 0.5
MAX_POINTS_DEFAULT = 12000

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def get_conn():
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c

def _f(x):
    try:
        return float(x)
    except Exception:
        return None

def _utc(ts):
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%H:%M:%S")
    except Exception:
        return "n/a"

def median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])

def iqr(xs: List[float]) -> float:
    if len(xs) < 4:
        return 0.0
    xs = sorted(xs)
    q1 = xs[len(xs) // 4]
    q3 = xs[(3 * len(xs)) // 4]
    return q3 - q1

# -----------------------------------------------------------------------------
# Unit inference
# -----------------------------------------------------------------------------
def infer_tmesh_to_seconds(rows):
    """
    infer scale so that:
      t_mesh_seconds = t_mesh_raw * scale
    """
    ratios = []
    by_node = {}

    for r in rows:
        n = r["node_id"]
        t = _f(r["created_at"])
        m = _f(r["t_mesh"])
        if n and t is not None and m is not None:
            by_node.setdefault(n, []).append((t, m))

    for pts in by_node.values():
        pts.sort()
        for i in range(1, len(pts)):
            dt = pts[i][0] - pts[i-1][0]
            dm = pts[i][1] - pts[i-1][1]
            if dt > 0 and dm > 0:
                ratios.append(dm / dt)

    r = median(ratios)
    if not r or r <= 0:
        return 1.0

    inv = 1.0 / r
    for snap in (1.0, 1e-3, 1e-6):
        if abs(math.log10(inv) - math.log10(snap)) < 0.25:
            return snap
    return inv

def mesh_offset_ms(t_created, t_mesh_raw, scale):
    return (t_mesh_raw * scale - t_created) * 1000.0

# -----------------------------------------------------------------------------
# Fetchers
# -----------------------------------------------------------------------------
def fetch_node_rows(window_s, limit):
    with get_conn() as c:
        cut = time.time() - window_s
        return c.execute("""
            SELECT node_id, created_at, t_mesh, offset, err_mesh_vs_wall
            FROM ntp_reference
            WHERE created_at >= ?
              AND peer_id IS NULL
            ORDER BY created_at ASC
            LIMIT ?
        """, (cut, limit)).fetchall()

def fetch_link_rows(window_s, limit):
    with get_conn() as c:
        cut = time.time() - window_s
        return c.execute("""
            SELECT node_id, peer_id, created_at, theta_ms, rtt_ms, sigma_ms
            FROM ntp_reference
            WHERE created_at >= ?
              AND peer_id IS NOT NULL
            ORDER BY created_at ASC
            LIMIT ?
        """, (cut, limit)).fetchall()

def fetch_controller_rows(window_s, limit):
    with get_conn() as c:
        cut = time.time() - window_s
        return c.execute("""
            SELECT node_id, created_at,
                   delta_desired_ms, delta_applied_ms, dt_s, slew_clipped
            FROM ntp_reference
            WHERE created_at >= ?
              AND (delta_desired_ms IS NOT NULL
                   OR delta_applied_ms IS NOT NULL
                   OR dt_s IS NOT NULL
                   OR slew_clipped IS NOT NULL)
            ORDER BY created_at ASC
            LIMIT ?
        """, (cut, limit)).fetchall()

# -----------------------------------------------------------------------------
# Mesh diagnostics (CORE)
# -----------------------------------------------------------------------------
def build_mesh_diag(window_s, bin_s, max_points):
    rows = fetch_node_rows(window_s, max_points)
    if not rows:
        return {"mesh_series": {}, "pairs": {}, "steps": {}}

    scale = infer_tmesh_to_seconds(rows)

    # bins[idx][node] -> list(mesh_offset_ms)
    bins = {}

    for r in rows:
        t = _f(r["created_at"])
        tm = _f(r["t_mesh"])
        n = r["node_id"]
        if t is None or tm is None or not n:
            continue
        idx = int(math.floor(t / bin_s))
        mo = mesh_offset_ms(t, tm, scale)
        bins.setdefault(idx, {}).setdefault(n, []).append(mo)

    mesh_series = {}
    steps = {}
    pairs = {}

    for idx in sorted(bins):
        bucket = bins[idx]
        if len(bucket) < 2:
            continue

        node_med = {n: median(v) for n, v in bucket.items() if median(v) is not None}
        if len(node_med) < 2:
            continue

        cons = median(list(node_med.values()))
        if cons is None:
            continue

        t_bin = (idx + 0.5) * bin_s

        eps = {}
        for n, m in node_med.items():
            e = m - cons
            eps[n] = e
            mesh_series.setdefault(n, []).append({"t": t_bin, "y": e})

        # steps
        for n, e in eps.items():
            prev = steps.setdefault(n, [])
            if prev:
                dy = e - prev[-1]["y"]
            else:
                dy = 0.0
            prev.append({"t": t_bin, "y": dy})

        # pairs
        ns = sorted(eps)
        for i in range(len(ns)):
            for j in range(i+1, len(ns)):
                pid = f"{ns[i]}-{ns[j]}"
                pairs.setdefault(pid, []).append({
                    "t": t_bin,
                    "y": eps[ns[i]] - eps[ns[j]]
                })

    return {
        "mesh_series": mesh_series,
        "step_series": steps,
        "pairs": pairs,
    }

# -----------------------------------------------------------------------------
# Link diagnostics
# -----------------------------------------------------------------------------
def build_link_diag(window_s, bin_s, max_points):
    rows = fetch_link_rows(window_s, max_points)
    bins = {}

    for r in rows:
        t = _f(r["created_at"])
        if t is None:
            continue
        idx = int(math.floor(t / bin_s))
        lid = f"{r['node_id']}->{r['peer_id']}"
        d = bins.setdefault(idx, {}).setdefault(lid, {"theta":[], "rtt":[], "sigma":[]})

        for k in ("theta_ms","rtt_ms","sigma_ms"):
            v = _f(r[k])
            if v is not None:
                d[k.split("_")[0]].append(v)

    out = {}
    latest_sigma = {}

    for idx in sorted(bins):
        t_bin = (idx + 0.5) * bin_s
        for lid, d in bins[idx].items():
            o = {"t": t_bin}
            if d["theta"]: o["theta_ms"] = median(d["theta"])
            if d["rtt"]:   o["rtt_ms"]   = median(d["rtt"])
            if d["sigma"]:
                o["sigma_ms"] = median(d["sigma"])
                latest_sigma[lid] = o["sigma_ms"]
            out.setdefault(lid, []).append(o)

    return {"links": out, "latest_sigma": latest_sigma}

# -----------------------------------------------------------------------------
# Controller diagnostics
# -----------------------------------------------------------------------------
def build_controller_diag(window_s, max_points):
    rows = fetch_controller_rows(window_s, max_points)

    # scale inference
    vals = []
    for r in rows:
        for k in ("delta_desired_ms","delta_applied_ms"):
            v = _f(r[k])
            if v is not None:
                vals.append(abs(v))
    scale = 1000.0 if median(vals) and median(vals) < 1.0 else 1.0

    out = {}
    for r in rows:
        n = r["node_id"]
        t = _f(r["created_at"])
        if not n or t is None:
            continue
        o = {"t": t}
        for k in ("delta_desired_ms","delta_applied_ms"):
            v = _f(r[k])
            if v is not None:
                o[k] = v * scale
        if r["dt_s"] is not None:
            o["dt_s"] = _f(r["dt_s"])
        if r["slew_clipped"] is not None:
            o["slew_clipped"] = int(r["slew_clipped"])
        out.setdefault(n, []).append(o)

    return {"controller": out, "delta_scale_to_ms": scale}

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/api/mesh_diag")
def api_mesh_diag():
    cfg = json.load(open(CFG_PATH)) if CFG_PATH.exists() else {}
    s = cfg.get("sync", {})
    return jsonify(build_mesh_diag(
        float(s.get("ui_window_s", WINDOW_S_DEFAULT)),
        float(s.get("ui_bin_s", BIN_S_DEFAULT)),
        int(s.get("ui_max_points", MAX_POINTS_DEFAULT)),
    ))

@app.route("/api/link_diag")
def api_link_diag():
    cfg = json.load(open(CFG_PATH)) if CFG_PATH.exists() else {}
    s = cfg.get("sync", {})
    return jsonify(build_link_diag(
        float(s.get("ui_window_s", WINDOW_S_DEFAULT)),
        float(s.get("ui_bin_s", BIN_S_DEFAULT)),
        int(s.get("ui_link_max_points", MAX_POINTS_DEFAULT)),
    ))

@app.route("/api/controller_diag")
def api_controller_diag():
    cfg = json.load(open(CFG_PATH)) if CFG_PATH.exists() else {}
    s = cfg.get("sync", {})
    return jsonify(build_controller_diag(
        float(s.get("ui_window_s", WINDOW_S_DEFAULT)),
        int(s.get("ui_ctrl_max_points", MAX_POINTS_DEFAULT)),
    ))

# -----------------------------------------------------------------------------
def ensure_db():
    Storage(str(DB_PATH))

if __name__ == "__main__":
    ensure_db()
    app.run(host="0.0.0.0", port=5000, debug=False)



# -----------------------------
# Template (Single Page)
# -----------------------------
TEMPLATE = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>MeshTime Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:0; padding: 1.4rem; background:#0f0f10; color:#eee; }
    h1,h2,h3 { margin:0; }
    .sub { margin-top:0.35rem; opacity:0.72; font-size:0.88rem; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .small { font-size:0.85rem; opacity:0.75; }
    .muted { opacity:0.7; }

    .card { background:#171718; border-radius:14px; padding:1rem 1.2rem; box-shadow: 0 0 0 1px rgba(255,255,255,0.05); }
    .row { display:flex; gap: 0.75rem; flex-wrap:wrap; align-items:center; }
    .sp { justify-content:space-between; }

    .pill { display:inline-block; padding:0.12rem 0.55rem; border-radius:999px; font-size:0.75rem; font-weight:750; }
    .ok   { background: rgba(46,204,113,0.14); color:#2ecc71; }
    .warn { background: rgba(241,196,15,0.14); color:#f1c40f; }
    .bad  { background: rgba(231,76,60,0.14); color:#e74c3c; }

    .grid { display:grid; grid-template-columns: minmax(0,1.25fr) minmax(0,2fr); gap: 1rem; align-items:start; margin-top: 1rem; }
    @media (max-width: 1200px) { .grid { grid-template-columns: minmax(0,1fr); } }

    .kpi-grid { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 0.75rem; margin-top: 0.75rem; }
    @media (max-width: 1000px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
    @media (max-width: 650px) { .kpi-grid { grid-template-columns: minmax(0,1fr); } }

    .kpi { border-radius: 12px; padding: 0.75rem; background: rgba(255,255,255,0.03); box-shadow: 0 0 0 1px rgba(255,255,255,0.03); }
    .kpi .title { font-size:0.85rem; opacity:0.75; }
    .kpi .value { font-size:1.2rem; font-weight:800; margin-top:0.2rem; }
    .kpi .hint { margin-top:0.25rem; font-size:0.82rem; opacity:0.68; }
    .kpi .reason { margin-top:0.35rem; font-size:0.85rem; opacity:0.75; }

    .plots { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin-top:0.75rem; }
    @media (max-width: 1200px) { .plots { grid-template-columns: minmax(0,1fr); } }

    table { width:100%; border-collapse: collapse; font-size:0.92rem; margin-top: 0.6rem; }
    th, td { padding: 0.35rem 0.5rem; text-align:left; vertical-align:top; }
    th { border-bottom: 1px solid rgba(255,255,255,0.15); font-weight:750; }
    tr:nth-child(even) td { background: rgba(255,255,255,0.02); }

    canvas { max-width:100%; }
    #meshCanvas { width:100%; height: 240px; background:#131313; border-radius:12px; margin-top:0.6rem; }

    .toggle { display:flex; gap:0.6rem; align-items:center; }
    .toggle input { transform: scale(1.2); }
    .hidden { display:none !important; }

    select {
      background:#101010; color:#eee; border: 1px solid rgba(255,255,255,0.18);
      border-radius: 10px; padding: 0.35rem 0.55rem;
    }
  </style>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@1.3.0/dist/chartjs-chart-matrix.min.js"></script>
</head>

<body>
  <div class="row sp">
    <div>
      <h1>MeshTime Dashboard</h1>
      <div class="sub">DB: <span class="mono">{{ db_path }}</span> · X-Achse: <b>created_at (Sink)</b></div>
    </div>

    <div class="row">
      <div class="toggle card" style="padding:0.55rem 0.85rem;">
        <input id="debugToggle" type="checkbox">
        <label for="debugToggle" class="small">Debug (Controller/Kalman)</label>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:1rem;">
    <div class="row sp">
      <div class="row">
        <span id="meshPill" class="pill warn">…</span>
        <div>
          <div style="font-weight:850;" id="meshLine">lade…</div>
          <div class="sub" id="meshMeta">lade…</div>
        </div>
      </div>
      <div class="row">
        <span class="small muted">Controller node:</span>
        <select id="ctrlNode"></select>
      </div>
    </div>

    <div class="kpi-grid" id="nodeKpis">
      <div class="kpi"><div class="title">Nodes</div><div class="value">…</div><div class="hint">—</div></div>
    </div>

    <div class="sub" style="margin-top:0.6rem;" id="offendersLine">lade…</div>
  </div>

  <div class="grid">
    <div>
      <div class="card">
        <h2 style="font-size:1.05rem;">Topologie</h2>
        <canvas id="meshCanvas"></canvas>
        <div class="sub">Node-Farbe = Konvergenz-Ampel. Link-Farbe/Dicke = σ (median im Conv-Window, max beider Richtungen).</div>
      </div>

      <div class="card" style="margin-top:1rem;">
        <h2 style="font-size:1.05rem;">Status (optimal mesh time)</h2>
        <div class="sub">
          Δ ist <b>Abweichung zur theoretisch optimalen Meshzeit</b> in ms.
          Hier ist “Meshzeit” bewusst als <span class="mono">(t_mesh - created_at)</span> definiert (Offset-artig), damit Sampling-Zeitverschiebungen nicht als Fehler erscheinen.
        </div>
        <table id="statusTable">
          <thead>
            <tr>
              <th>Node</th>
              <th>State</th>
              <th>mesh_err_now</th>
              <th>rate</th>
              <th>age</th>
              <th>reason</th>
            </tr>
          </thead>
          <tbody><tr><td colspan="6" class="small">lade…</td></tr></tbody>
        </table>
      </div>

      <div class="card debugOnly" style="margin-top:1rem;">
        <h2 style="font-size:1.05rem;">Konvergenz-Regeln (Debug)</h2>
        <div class="sub mono" id="thrLine">lade…</div>
      </div>
    </div>

    <div>
      <div class="card">
        <h2 style="font-size:1.05rem;">Centered Mesh-Time Offset: ε(node,t) (ms)</h2>
        <div class="sub">
          ε basiert auf <span class="mono">mesh_offset_ms=(t_mesh - created_at)*1000</span> und ist pro Bin um den Median zentriert.
          Das ist der Plot, der “wirklich stimmt”.
        </div>
        <canvas id="meshChart" height="170"></canvas>

        <div class="plots">
          <div>
            <h3 class="small">Pairwise: ε_i − ε_j (ms)</h3>
            <canvas id="pairChart" height="150"></canvas>
          </div>
          <div>
            <h3 class="small">Mesh-Error Step: Δε pro Node (ms)</h3>
            <canvas id="stepChart" height="150"></canvas>
          </div>
          <div>
            <h3 class="small">Pair Stability: σ ≈ 0.741·IQR(ε_i − ε_j) (ms)</h3>
            <canvas id="stabilityBar" height="150"></canvas>
          </div>
          <div>
            <h3 class="small">Heatmap: |ε_i − ε_j| (ms) (binned)</h3>
            <canvas id="heatmapChart" height="150"></canvas>
          </div>

          <div>
            <h3 class="small">offset (ms) (legacy)</h3>
            <canvas id="legacyOffset" height="150"></canvas>
          </div>
          <div>
            <h3 class="small">err_mesh_vs_wall (ms) (legacy)</h3>
            <canvas id="legacyErr" height="150"></canvas>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top:1rem;">
        <h2 style="font-size:1.05rem;">Link Quality (θ / RTT / σ)</h2>
        <div class="sub muted">Link-ID ist <span class="mono">A-&gt;B</span> (gerichtet). θ ist signiert.</div>

        <div class="plots">
          <div>
            <h3 class="small">θ pro Link (ms)</h3>
            <canvas id="thetaChart" height="150"></canvas>
          </div>
          <div>
            <h3 class="small">RTT pro Link (ms)</h3>
            <canvas id="rttChart" height="150"></canvas>
          </div>
          <div>
            <h3 class="small">σ pro Link (latest) (ms)</h3>
            <canvas id="sigmaBar" height="150"></canvas>
          </div>
          <div class="small">
            <b>Interpretation:</b><br>
            σ hoch → Link noisy/unstable → Ampel wird gelb/rot.<br>
            RTT spiky → Medium/Queueing. θ spiky → Asymmetrie/Peer/Bootstrap.
          </div>
        </div>
      </div>

      <div class="card debugOnly" style="margin-top:1rem;">
        <h2 style="font-size:1.05rem;">Controller / Kalman Diagnose (Debug)</h2>
        <div class="sub">delta_desired vs delta_applied, dt, slew_clipped → Ursache/Wirkung</div>

        <div class="plots">
          <div>
            <h3 class="small">delta_desired_ms</h3>
            <canvas id="deltaDesired" height="150"></canvas>
          </div>
          <div>
            <h3 class="small">delta_applied_ms</h3>
            <canvas id="deltaApplied" height="150"></canvas>
          </div>
          <div>
            <h3 class="small">dt_s</h3>
            <canvas id="dtChart" height="150"></canvas>
          </div>
          <div>
            <h3 class="small">slew_clipped (0/1)</h3>
            <canvas id="slewChart" height="150"></canvas>
          </div>
        </div>
      </div>

      <div class="small muted" style="margin-top:1rem;">
        Reality check: Wenn es “stabil aussieht”, aber Ampel ist gelb/rot → schau Reason: stale? correction? stable_links? slew_clipped?
      </div>
    </div>
  </div>

<script>
  const colors = ['#2ecc71','#3498db','#f1c40f','#e74c3c','#9b59b6','#1abc9c','#e67e22'];

  let meshChart, pairChart, stepChart, stabilityBar, heatmapChart;
  let thetaChart, rttChart, sigmaBar;
  let deltaDesiredChart, deltaAppliedChart, dtLineChart, slewLineChart;
  let legacyOffsetChart, legacyErrChart;

  function pillClass(state){
    if(state === 'GREEN') return 'pill ok';
    if(state === 'RED') return 'pill bad';
    return 'pill warn';
  }

  function setMeshHeader(state, line, meta){
    const el = document.getElementById('meshPill');
    el.className = pillClass(state);
    el.textContent = state || '…';
    document.getElementById('meshLine').textContent = line || '';
    document.getElementById('meshMeta').textContent = meta || '';
  }

  async function fetchJson(url){
    const resp = await fetch(url);
    let data = null;
    try { data = await resp.json(); } catch(e) { data = { error: "invalid json" }; }
    if(!resp.ok) throw new Error(`${url}: ${data.error || 'unknown error'}`);
    return data;
  }

  function mkLine(ctx, yLabel){
    return new Chart(ctx, {
      type:'line',
      data:{datasets:[]},
      options:{
        responsive:true, animation:false,
        scales:{
          x:{type:'time', ticks:{color:'#aaa'}, grid:{color:'rgba(255,255,255,0.06)'}},
          y:{ticks:{color:'#aaa'}, grid:{color:'rgba(255,255,255,0.06)'}, title:{display:true, text:yLabel, color:'#aaa'}}
        },
        plugins:{ legend:{labels:{color:'#eee'}} },
        elements:{ point:{radius:0}, line:{tension:0.06} }
      }
    });
  }

  function mkBar(ctx, label, yLabel){
    return new Chart(ctx, {
      type:'bar',
      data:{labels:[], datasets:[{label:label, data:[]}]},
      options:{
        responsive:true, animation:false,
        scales:{
          x:{ticks:{color:'#aaa'}, grid:{display:false}},
          y:{ticks:{color:'#aaa'}, grid:{color:'rgba(255,255,255,0.06)'}, title:{display:true, text:yLabel, color:'#aaa'}}
        },
        plugins:{ legend:{labels:{color:'#eee'}} }
      }
    });
  }

  function mkHeatmap(ctx){
    return new Chart(ctx, {
      type:'matrix',
      data:{datasets:[{
        label:'Heatmap',
        data:[],
        borderWidth:0,
        backgroundColor:(context)=>{
          const raw = context.raw;
          if(!raw || typeof raw.v !== 'number') return 'rgba(0,0,0,0)';
          const maxV = context.chart._maxV || 1;
          const ratio = Math.min(1, raw.v / maxV);
          const r = Math.round(255 * ratio);
          const g = Math.round(255 * (1 - ratio));
          return `rgba(${r},${g},150,0.85)`;
        },
        width:(context)=>{
          const area = context.chart.chartArea || {};
          const nBins = context.chart._nBins || 1;
          const w = (area.right - area.left) / nBins;
          return (Number.isFinite(w) && w > 0) ? w : 10;
        },
        height:(context)=>{
          const area = context.chart.chartArea || {};
          const cats = context.chart._cats || ['pair'];
          const h = (area.bottom - area.top) / (cats.length || 1);
          return (Number.isFinite(h) && h > 0) ? h : 10;
        }
      }]},
      options:{
        responsive:true, animation:false,
        scales:{
          x:{type:'time', ticks:{color:'#aaa'}, grid:{color:'rgba(255,255,255,0.06)'}},
          y:{
            type:'linear',
            ticks:{color:'#aaa', callback:function(v){
              const cats = this.chart._cats || [];
              const idx = Math.round(v);
              return cats[idx] || '';
            }},
            grid:{color:'rgba(255,255,255,0.06)'}
          }
        },
        plugins:{
          legend:{labels:{color:'#eee'}},
          tooltip:{callbacks:{label:(ctx)=>{
            const cats = ctx.chart._cats || [];
            const pid = cats[Math.round(ctx.raw.y)] || '?';
            const t = new Date(ctx.raw.x);
            return `${pid} @ ${t.toLocaleTimeString()} : |Δ|=${ctx.raw.v.toFixed(2)} ms`;
          }}}
        }
      }
    });
  }

  function updateLine(chart, seriesMap, field, labelPrefix=''){
    const ids = Object.keys(seriesMap||{}).sort();
    chart.data.datasets = [];
    ids.forEach((id, idx)=>{
      const c = colors[idx % colors.length];
      const pts = (seriesMap[id]||[])
        .filter(p => p && p[field] !== null && p[field] !== undefined && Number.isFinite(p[field]))
        .map(p => ({x:new Date(p.t_wall*1000), y:p[field]}));
      chart.data.datasets.push({label:`${labelPrefix}${id}`, data:pts, borderColor:c, borderWidth:1.6});
    });
    chart.update();
  }

  function updatePairs(chart, pairs){
    const ids = Object.keys(pairs||{}).sort();
    chart.data.datasets = [];
    ids.forEach((id, idx)=>{
      const c = colors[idx % colors.length];
      const pts = (pairs[id]||[]).map(p => ({x:new Date(p.t_wall*1000), y:p.delta_ms}));
      chart.data.datasets.push({label:id, data:pts, borderColor:c, borderWidth:1.6});
    });
    chart.update();
  }

  function updateBarFromMap(chart, mapObj, field){
    const ids = Object.keys(mapObj||{}).sort();
    const labels=[], data=[];
    ids.forEach(id=>{
      const v = mapObj[id] ? mapObj[id][field] : null;
      if(v !== null && v !== undefined && Number.isFinite(v)){
        labels.push(id); data.push(v);
      }
    });
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update();
  }

  function updateSigmaBar(chart, latest){
    const ids = Object.keys(latest||{}).sort();
    const labels=[], data=[];
    ids.forEach(id=>{
      const v = latest[id];
      if(v !== null && v !== undefined && Number.isFinite(v)){
        labels.push(id); data.push(v);
      }
    });
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update();
  }

  function updateHeatmap(chart, heatmap){
    const data = (heatmap && heatmap.data) || [];
    if(!data.length){
      chart.data.datasets[0].data = [];
      chart._cats = [];
      chart._nBins = 1;
      chart._maxV = 1;
      chart.update();
      return;
    }
    const cats = Array.from(new Set(data.map(d=>d.pair))).sort();
    const idx = new Map(cats.map((c,i)=>[c,i]));
    let maxV = 0;
    const matrix = data.map(d=>{
      const v = Math.abs(d.value);
      if(v > maxV) maxV = v;
      return {x:new Date(d.t_bin*1000), y:(idx.get(d.pair) ?? 0), v:v};
    });
    chart.data.datasets[0].data = matrix;
    chart._cats = cats;
    chart._nBins = heatmap.n_bins || 1;
    chart._maxV = maxV || 1;
    chart.update();
  }

  function renderNodeKpis(nodes){
    const wrap = document.getElementById('nodeKpis');
    wrap.innerHTML = '';
    (nodes||[]).forEach(n=>{
      const st = n.state || 'YELLOW';
      const off = (n.mesh_err_now_ms==null) ? 'n/a' : `${n.mesh_err_now_ms.toFixed(2)} ms`;
      const age = (n.age_s==null) ? 'n/a' : `${n.age_s.toFixed(1)} s`;
      const rate = (n.mesh_rate_ms_s==null) ? '—' : `${n.mesh_rate_ms_s.toFixed(3)} ms/s`;
      const corr = (n.med_abs_delta_applied_ms==null) ? 'n/a' : `${n.med_abs_delta_applied_ms.toFixed(2)} ms`;
      const clip = (n.slew_clip_rate==null) ? '—' : `${(n.slew_clip_rate*100).toFixed(0)}%`;
      const sl = `${n.stable_links ?? 0}/${n.k_stable_links ?? 1}`;

      const div = document.createElement('div');
      div.className = 'kpi';
      div.innerHTML = `
        <div class="row sp">
          <div class="title">Node ${n.node_id}</div>
          <span class="${pillClass(st)}">${st}</span>
        </div>
        <div class="value">${off}</div>
        <div class="hint">rate ${rate} · age ${age}</div>
        <div class="hint">|Δapplied|med ${corr} · clip ${clip} · stable_links ${sl}</div>
        <div class="reason">${n.reason || ''}</div>
      `;
      wrap.appendChild(div);
    });
  }

  function renderStatusTable(nodes){
    const tb = document.querySelector('#statusTable tbody');
    tb.innerHTML = '';
    if(!nodes || !nodes.length){
      tb.innerHTML = `<tr><td colspan="6" class="small">Keine Daten.</td></tr>`;
      return;
    }
    nodes.forEach(n=>{
      const st = n.state || 'YELLOW';
      const pill = pillClass(st);
      const err = (n.mesh_err_now_ms==null) ? 'n/a' : `${n.mesh_err_now_ms.toFixed(2)} ms`;
      const rate = (n.mesh_rate_ms_s==null) ? '—' : `${n.mesh_rate_ms_s.toFixed(3)} ms/s`;
      const age = (n.age_s==null) ? 'n/a' : `${n.age_s.toFixed(1)} s`;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${n.node_id}</td>
        <td><span class="${pill}">${st}</span></td>
        <td>${err}</td>
        <td>${rate}</td>
        <td>${age}</td>
        <td class="small">${n.reason || ''}</td>
      `;
      tb.appendChild(tr);
    });
  }

  function drawMesh(canvas, topo, nodeStates, linkSigmaMed){
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);

    const nodes = topo.nodes || [];
    const links = topo.links || [];
    if(!nodes.length){
      ctx.fillStyle = '#777';
      ctx.font = '12px system-ui';
      ctx.fillText('Keine Nodes in config/nodes.json gefunden.', 10, 20);
      return;
    }

    const n = nodes.length;
    const radius = Math.min(w,h) * 0.35;
    const cx = w/2, cy = h/2;

    const pos = {};
    nodes.forEach((node, i)=>{
      const a = (2*Math.PI*i)/n - Math.PI/2;
      pos[node.id] = {x: cx + radius*Math.cos(a), y: cy + radius*Math.sin(a)};
    });

    function sigForUndirected(a,b){
      const s1 = linkSigmaMed[`${a}->${b}`];
      const s2 = linkSigmaMed[`${b}->${a}`];
      if(s1==null && s2==null) return null;
      if(s1==null) return s2;
      if(s2==null) return s1;
      return Math.max(s1,s2);
    }

    links.forEach(l=>{
      const a = pos[l.source], b = pos[l.target];
      if(!a || !b) return;

      const sig = sigForUndirected(l.source, l.target);
      let stroke = 'rgba(255,255,255,0.20)';
      let lw = 1.0;
      if(sig != null){
        if(sig <= 2.0){ stroke = 'rgba(46,204,113,0.55)'; lw = 2.6; }
        else if(sig <= 5.0){ stroke = 'rgba(241,196,15,0.55)'; lw = 2.1; }
        else { stroke = 'rgba(231,76,60,0.55)'; lw = 2.1; }
      }

      ctx.strokeStyle = stroke;
      ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
    });

    nodes.forEach(node=>{
      const p = pos[node.id]; if(!p) return;
      const st = nodeStates[node.id] || 'YELLOW';
      const isRoot = !!node.is_root;
      const r = isRoot ? 20 : 16;

      let fill = 'rgba(241,196,15,0.14)', stroke = '#f1c40f';
      if(st === 'GREEN'){ fill = 'rgba(46,204,113,0.18)'; stroke = '#2ecc71'; }
      if(st === 'RED'){ fill = 'rgba(231,76,60,0.18)'; stroke = '#e74c3c'; }

      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,2*Math.PI);
      ctx.fillStyle = fill; ctx.fill();
      ctx.lineWidth = isRoot ? 3.0 : 2.0;
      ctx.strokeStyle = stroke; ctx.stroke();

      ctx.fillStyle = '#ecf0f1';
      ctx.font = '12px system-ui';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(node.id, p.x, p.y);
    });
  }

  function applyDebugVisibility(){
    const on = document.getElementById('debugToggle').checked;
    document.querySelectorAll('.debugOnly').forEach(el=>{
      el.classList.toggle('hidden', !on);
    });
  }

  function updateControllerCharts(ctrl, selectedNode){
    const byNode = ctrl || {};
    const pts = byNode[selectedNode] || [];

    function mkDataset(field, label){
      const data = pts
        .filter(p => p && p[field] !== null && p[field] !== undefined && Number.isFinite(p[field]))
        .map(p => ({x:new Date(p.t_wall*1000), y:p[field]}));
      return [{label: label, data: data, borderColor: '#2ecc71', borderWidth: 1.6}];
    }

    deltaDesiredChart.data.datasets = mkDataset('delta_desired_ms', 'selected node');
    deltaAppliedChart.data.datasets = mkDataset('delta_applied_ms', 'selected node');
    dtLineChart.data.datasets       = mkDataset('dt_s', 'selected node');
    slewLineChart.data.datasets     = mkDataset('slew_clipped', 'selected node');

    deltaDesiredChart.update();
    deltaAppliedChart.update();
    dtLineChart.update();
    slewLineChart.update();
  }

  function initCharts(){
    meshChart      = mkLine(document.getElementById('meshChart').getContext('2d'), 'ms');
    pairChart      = mkLine(document.getElementById('pairChart').getContext('2d'), 'ms');
    stepChart      = mkLine(document.getElementById('stepChart').getContext('2d'), 'ms');
    stabilityBar   = mkBar(document.getElementById('stabilityBar').getContext('2d'), 'σ_pair', 'ms');
    heatmapChart   = mkHeatmap(document.getElementById('heatmapChart').getContext('2d'));

    legacyOffsetChart = mkLine(document.getElementById('legacyOffset').getContext('2d'), 'ms');
    legacyErrChart    = mkLine(document.getElementById('legacyErr').getContext('2d'), 'ms');

    thetaChart     = mkLine(document.getElementById('thetaChart').getContext('2d'), 'ms');
    rttChart       = mkLine(document.getElementById('rttChart').getContext('2d'), 'ms');
    sigmaBar       = mkBar(document.getElementById('sigmaBar').getContext('2d'), 'σ latest', 'ms');

    deltaDesiredChart = mkLine(document.getElementById('deltaDesired').getContext('2d'), 'ms');
    deltaAppliedChart = mkLine(document.getElementById('deltaApplied').getContext('2d'), 'ms');
    dtLineChart       = mkLine(document.getElementById('dtChart').getContext('2d'), 's');
    slewLineChart     = mkLine(document.getElementById('slewChart').getContext('2d'), '0/1');
  }

  async function refresh(){
    try{
      const topo = await fetchJson('/api/topology');
      const [ov, mesh, link, ctrl] = await Promise.all([
        fetchJson('/api/overview'),
        fetchJson('/api/mesh_diag'),
        fetchJson('/api/link_diag'),
        fetchJson('/api/controller_diag'),
      ]);

      // header
      const M = ov.mesh || {};
      setMeshHeader(
        M.state || 'YELLOW',
        `${M.state || '…'}: ${M.reason || ''}`,
        `now=${M.now_utc || 'n/a'} · conv_window=${M.conv_window_s || '?'}s · t_mesh_to_seconds=${(M.t_mesh_to_seconds ?? '?')} · consensus_offset_ms=${(M.consensus_now_offset_ms ?? 'n/a')}`
      );

      const T = (M.thresholds || {});
      document.getElementById('thrLine').textContent =
        `fresh≤${T.fresh_s ?? '?'}s  |Δapplied|med≤${T.delta_applied_med_ms ?? '?'}ms  linkσmed≤${T.link_sigma_med_ms ?? '?'}ms  clip≤${Math.round((T.slew_clip_rate ?? 0)*100)}%  K_stable_links=${T.k_stable_links ?? '?'}  warmup≥${T.warmup_min_samples ?? '?'} samples`;

      // offenders
      const off = ov.offenders || {};
      document.getElementById('offendersLine').textContent =
        `worst correction: ${off.worst_node_correction || '—'} · stalest: ${off.stalest_node || '—'} · most clipped: ${off.most_slew_clipped || '—'} · worst link σ: ${off.worst_link_sigma || '—'}`;

      // node tiles + status table
      renderNodeKpis(ov.nodes || []);
      renderStatusTable(ov.nodes || []);

      // topology
      const nodeStates = {};
      (ov.nodes || []).forEach(n => { nodeStates[n.node_id] = n.state; });

      const meshCanvas = document.getElementById('meshCanvas');
      meshCanvas.width = meshCanvas.clientWidth;
      meshCanvas.height = meshCanvas.clientHeight;
      drawMesh(meshCanvas, topo, nodeStates, ov.link_sigma_med || {});

      // plots
      updateLine(meshChart, mesh.mesh_series || {}, "mesh_err_ms", "Node ");
      updatePairs(pairChart, mesh.pairs || {});
      updateLine(stepChart, mesh.step_series || {}, "step_ms", "Node ");
      updateBarFromMap(stabilityBar, mesh.stability || {}, "sigma_ms");
      updateHeatmap(heatmapChart, mesh.heatmap || {data:[], n_bins:0});

      // legacy
      const L = (mesh.legacy || {});
      updateLine(legacyOffsetChart, L.offset || {}, "offset_ms", "Node ");
      updateLine(legacyErrChart, L.err_mesh_vs_wall || {}, "err_ms", "Node ");

      // link plots
      const links = link.links || {};
      updateLine(thetaChart, links, "theta_ms", "");
      updateLine(rttChart, links, "rtt_ms", "");
      updateSigmaBar(sigmaBar, link.latest_sigma || {});

      // controller dropdown
      const ctrlMap = ctrl.controller || {};
      const sel = document.getElementById('ctrlNode');
      const nodeIds = Object.keys(ctrlMap).sort();

      if(sel.options.length === 0){
        nodeIds.forEach(n=>{
          const o = document.createElement('option');
          o.value = n;
          o.textContent = n;
          sel.appendChild(o);
        });
        if(nodeIds.length) sel.value = nodeIds[0];
      }
      const chosen = sel.value || (nodeIds[0] || '');
      if(chosen){
        updateControllerCharts(ctrlMap, chosen);
      }

    } catch(e){
      console.error('refresh failed:', e);
    }
  }

  window.addEventListener('load', ()=>{
    initCharts();
    document.getElementById('debugToggle').addEventListener('change', applyDebugVisibility);
    document.getElementById('ctrlNode').addEventListener('change', ()=>refresh());
    applyDebugVisibility();
    refresh();
    setInterval(refresh, 1000);
  });
</script>

</body>
</html>
"""


def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Storage(str(DB_PATH))


if __name__ == "__main__":
    ensure_db()
    print("Starting MeshTime Dashboard on http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)
