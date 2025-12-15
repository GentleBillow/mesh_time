# -*- coding: utf-8 -*-
# mesh/web_ui.py
#
# MeshTime Dashboard — Single Page, Operator-first
#
# HARD RULES:
# - created_at (sink clock) is the ONLY x-axis
# - Mesh-Time is first-class: t_mesh(node) - consensus(t)
# - Deltas are diagnosis, never truth
# - Units are explicit: ms / ms/s / s
# - Must never crash on None

from __future__ import annotations

import json
import math
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

from flask import Flask, jsonify, render_template_string

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "mesh_data.sqlite"
CFG_PATH = BASE_DIR / "config" / "nodes.json"

from mesh.storage import Storage  # ensure schema

app = Flask(__name__)

# ----------------------------
# Tuning
# ----------------------------
UI_WINDOW_S = 120
UI_BIN_S = 0.5
MAX_POINTS = 8000

CONV_WINDOW_S = 30.0
T_DELTA_APPLIED_MS = 0.5
T_SIGMA_MS = 2.0
T_FRESH_MIN_S = 3.0


# ----------------------------
# Helpers
# ----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _f(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


# ----------------------------
# Data access
# ----------------------------
def fetch_node_rows(window_s: float) -> List[sqlite3.Row]:
    conn = get_conn()
    try:
        cut = time.time() - window_s
        return conn.execute(
            """
            SELECT created_at, node_id, t_mesh, delta_desired_ms,
                   delta_applied_ms, dt_s, slew_clipped
            FROM ntp_reference
            WHERE peer_id IS NULL
              AND created_at >= ?
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (cut, MAX_POINTS),
        ).fetchall()
    finally:
        conn.close()


def fetch_link_rows(window_s: float) -> List[sqlite3.Row]:
    conn = get_conn()
    try:
        cut = time.time() - window_s
        return conn.execute(
            """
            SELECT created_at, node_id, peer_id, theta_ms, sigma_ms
            FROM ntp_reference
            WHERE peer_id IS NOT NULL
              AND created_at >= ?
            ORDER BY created_at ASC
            """,
            (cut,),
        ).fetchall()
    finally:
        conn.close()


# ----------------------------
# Aggregation
# ----------------------------
def build_mesh_timeseries(window_s: float, bin_s: float):
    rows = fetch_node_rows(window_s)
    if not rows:
        return {}

    t0 = min(_f(r["created_at"]) for r in rows if _f(r["created_at"]) is not None)

    # bins[bin][node] = last t_mesh
    bins: Dict[int, Dict[str, float]] = {}

    for r in rows:
        t = _f(r["created_at"])
        tm = _f(r["t_mesh"])
        if t is None or tm is None:
            continue

        idx = int((t - t0) / bin_s)
        bins.setdefault(idx, {})[r["node_id"]] = tm * 1000.0  # → ms ONCE

    series: Dict[str, List[Dict[str, float]]] = {}

    for idx in sorted(bins):
        bucket = bins[idx]
        if len(bucket) < 2:
            continue

        consensus = _median(list(bucket.values()))
        t_bin = t0 + (idx + 0.5) * bin_s

        for node, tm_ms in bucket.items():
            series.setdefault(node, []).append({
                "t": t_bin,
                "mesh_ms": tm_ms - consensus
            })

    return series


def build_controller_timeseries(window_s: float):
    rows = fetch_node_rows(window_s)
    out: Dict[str, List[Dict[str, float]]] = {}

    for r in rows:
        t = _f(r["created_at"])
        if t is None:
            continue

        node = r["node_id"]
        d = {"t": t}

        for k in ["delta_desired_ms", "delta_applied_ms", "dt_s"]:
            v = _f(r[k])
            if v is not None:
                d[k] = v

        sc = r["slew_clipped"]
        if sc is not None:
            d["slew_clipped"] = int(sc)

        out.setdefault(node, []).append(d)

    return out


def build_link_table(window_s: float):
    rows = fetch_link_rows(window_s)
    by_link: Dict[str, Dict[str, List[float]]] = {}

    for r in rows:
        lid = f"{r['node_id']}→{r['peer_id']}"
        by_link.setdefault(lid, {"theta": [], "sigma": []})

        th = _f(r["theta_ms"])
        sg = _f(r["sigma_ms"])

        if th is not None:
            by_link[lid]["theta"].append(th)
        if sg is not None:
            by_link[lid]["sigma"].append(sg)

    table = []
    for lid, vals in sorted(by_link.items()):
        table.append({
            "link": lid,
            "theta_med": _median(vals["theta"]),
            "sigma_med": _median(vals["sigma"]),
        })

    return table


# ----------------------------
# API
# ----------------------------
@app.route("/api/data")
def api_data():
    return jsonify({
        "mesh": build_mesh_timeseries(UI_WINDOW_S, UI_BIN_S),
        "controller": build_controller_timeseries(UI_WINDOW_S),
        "links": build_link_table(UI_WINDOW_S),
    })


# ----------------------------
# UI
# ----------------------------
TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>MeshTime</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
<style>
body { background:#111; color:#eee; font-family:system-ui; margin:0; padding:1rem; }
h2 { margin-top:1.2rem; }
canvas { background:#161616; border-radius:12px; }
.debug { margin-top:2rem; opacity:0.95; }
.hidden { display:none; }
</style>
</head>

<body>
<h1>MeshTime</h1>

<h2>t_mesh(node) − Konsens über Zeit</h2>
<canvas id="meshChart" height="260"></canvas>

<h2>Links (Median im Fenster)</h2>
<table id="linkTable"></table>

<div class="debug hidden" id="debug">
<h2>Controller (Debug)</h2>
<canvas id="d_des" height="120"></canvas>
<canvas id="d_app" height="120"></canvas>
<canvas id="dt" height="120"></canvas>
<canvas id="clip" height="120"></canvas>
</div>

<script>
let meshChart, dDes, dApp, dtChart, clipChart;

function mkLine(ctx, label){
  return new Chart(ctx,{
    type:'line',
    data:{datasets:[]},
    options:{
      animation:false,
      scales:{
        x:{type:'time'},
        y:{title:{display:true,text:label}}
      },
      elements:{point:{radius:0}}
    }
  });
}

async function refresh(){
  const data = await (await fetch('/api/data')).json();

  // ---- Mesh plot ----
  meshChart.data.datasets = [];
  Object.entries(data.mesh).forEach(([node,pts],i)=>{
    meshChart.data.datasets.push({
      label:node,
      data:pts.map(p=>({x:p.t*1000,y:p.mesh_ms})),
      borderWidth:1.5
    });
  });
  meshChart.update();

  // ---- Links ----
  const tbl = document.getElementById('linkTable');
  tbl.innerHTML = '<tr><th>Link</th><th>θ med (ms)</th><th>σ med (ms)</th></tr>';
  data.links.forEach(r=>{
    tbl.innerHTML += `<tr><td>${r.link}</td><td>${r.theta_med?.toFixed(2)||'—'}</td><td>${r.sigma_med?.toFixed(2)||'—'}</td></tr>`;
  });

  // ---- Controller ----
  function upd(chart, key){
    chart.data.datasets=[];
    Object.entries(data.controller).forEach(([node,pts])=>{
      const d=pts.filter(p=>p[key]!=null).map(p=>({x:p.t*1000,y:p[key]}));
      if(d.length) chart.data.datasets.push({label:node,data:d});
    });
    chart.update();
  }

  upd(dDes,'delta_desired_ms');
  upd(dApp,'delta_applied_ms');
  upd(dtChart,'dt_s');
  upd(clipChart,'slew_clipped');
}

window.onload=()=>{
  meshChart = mkLine(document.getElementById('meshChart'),'ms');
  dDes = mkLine(document.getElementById('d_des'),'delta_desired_ms');
  dApp = mkLine(document.getElementById('d_app'),'delta_applied_ms');
  dtChart = mkLine(document.getElementById('dt'),'dt_s');
  clipChart = mkLine(document.getElementById('clip'),'slew_clipped');

  refresh();
  setInterval(refresh,2000);
};
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(TEMPLATE)


if __name__ == "__main__":
    Storage(str(DB_PATH))
    app.run(host="0.0.0.0", port=5000, debug=False)
