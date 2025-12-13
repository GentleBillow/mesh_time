# -*- coding: utf-8 -*-
# mesh/web_ui.py
# MeshTime Web-UI: status + bin-synced pairwise ΔOffset + robust jitter (IQR/MAD) + heatmap + topology
# + Link metrics (theta/rtt/sigma) from ntp_reference(peer_id, theta_ms, rtt_ms, sigma_ms)
#
# IMPORTANT FIX:
#   Use created_at (sink clock, e.g. node C) as the global X-axis for binning and plots.
#   Never use sender t_wall as a global time axis in unrooted mesh.

import json
import math
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

from flask import Flask, render_template_string, jsonify, make_response

BASE_DIR = Path(__file__).resolve().parent.parent  # /home/pi/mesh_time
DB_PATH = BASE_DIR / "mesh_data.sqlite"
CFG_PATH = BASE_DIR / "config" / "nodes.json"

from mesh.storage import Storage  # ensures schema

app = Flask(__name__)

# -----------------------------
# Tuning
# -----------------------------
WINDOW_SECONDS_DEFAULT = 600        # 10 Minuten
MAX_POINTS_DEFAULT = 6000           # DB read limit
BIN_S_DEFAULT = 0.5                 # 500ms Bin
HEATMAP_MAX_BINS = 40               # max bins in heatmap (UI)
JITTER_MIN_SAMPLES = 10             # min bins per pair for robust stats

# Link-metrics tuning
LINK_MAX_POINTS_DEFAULT = 8000      # allow more link rows
LINK_MIN_SAMPLES = 8               # min bins for summaries


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _json_error(msg: str, status: int = 500):
    return make_response(jsonify({"error": msg}), status)


def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def load_config() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        return {}
    with CFG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_topology() -> Dict[str, Any]:
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


def _table_cols(conn: sqlite3.Connection, table: str) -> set:
    cur = conn.cursor()
    return {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}


# ------------------------------------------------------------
# Robust statistics (IQR/MAD)
# ------------------------------------------------------------

def _quantile_sorted(xs: List[float], q: float) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return xs[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def robust_iqr_ms(values_ms: List[float]) -> float:
    if not values_ms:
        return 0.0
    xs = sorted(values_ms)
    q1 = _quantile_sorted(xs, 0.25)
    q3 = _quantile_sorted(xs, 0.75)
    return float(q3 - q1)


def robust_mad_ms(values_ms: List[float]) -> float:
    if not values_ms:
        return 0.0
    xs = sorted(values_ms)
    med = _quantile_sorted(xs, 0.50)
    abs_dev = sorted([abs(v - med) for v in values_ms])
    mad = _quantile_sorted(abs_dev, 0.50)
    return float(mad)


# ------------------------------------------------------------
# Status snapshot (NO t_mesh-based "reference", only offsets)
# ------------------------------------------------------------

def get_status_snapshot(reference_node: str = "C") -> Dict[str, Any]:
    """
    Status is purely based on the newest offset per node.
    Δ vs Ref is computed from offset differences, not t_mesh.
    created_at is used for age (sink clock).
    """
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

    now = time.time()
    rows_out: List[Dict[str, Any]] = []

    ref_offset_ms: Optional[float] = None
    for r in last_by_node:
        if r["node_id"] == reference_node:
            try:
                ref_offset_ms = float(r["offset"]) * 1000.0
            except Exception:
                ref_offset_ms = None

    for r in last_by_node:
        node_id = r["node_id"]

        offset_ms = None
        created_at = None
        t_wall = None

        try:
            offset_ms = float(r["offset"]) * 1000.0
        except Exception:
            pass

        try:
            created_at = float(r["created_at"])
        except Exception:
            created_at = now

        try:
            t_wall = float(r["t_wall"])
        except Exception:
            t_wall = None

        age_s = (now - created_at) if created_at is not None else None

        delta_vs_ref_ms = None
        if (offset_ms is not None) and (ref_offset_ms is not None):
            delta_vs_ref_ms = offset_ms - ref_offset_ms

        t_disp = created_at
        rows_out.append({
            "node_id": node_id,
            "t_mesh": t_disp,
            "t_mesh_utc": datetime.utcfromtimestamp(t_disp).strftime("%Y-%m-%d %H:%M:%S") if t_disp is not None else "n/a",
            "offset_ms": offset_ms,
            "t_wall": t_wall,
            "t_wall_utc": datetime.utcfromtimestamp(t_wall).strftime("%Y-%m-%d %H:%M:%S") if t_wall is not None else "n/a",
            "age_s": age_s,
            "delta_vs_ref_ms": delta_vs_ref_ms,
        })

    return {"rows": rows_out, "reference_node": reference_node}


# ------------------------------------------------------------
# Bin-synced timeseries (use created_at as global axis!)
# ------------------------------------------------------------

def get_ntp_timeseries(
    window_seconds: int = WINDOW_SECONDS_DEFAULT,
    max_points: int = MAX_POINTS_DEFAULT,
    bin_s: float = BIN_S_DEFAULT
) -> Dict[str, Any]:

    conn = get_conn()
    cur = conn.cursor()

    cutoff = time.time() - float(window_seconds)

    rows = cur.execute(
        """
        SELECT node_id, offset, created_at
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
            "jitter": {},
            "heatmap": {"data": [], "n_bins": 0},
            "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"},
        }

    t_list: List[float] = []
    for r in rows:
        try:
            t_list.append(float(r["created_at"]))
        except Exception:
            pass

    if not t_list:
        return {
            "series": {},
            "pairs": {},
            "jitter": {},
            "heatmap": {"data": [], "n_bins": 0},
            "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"},
        }

    t_min = min(t_list)
    t_max = max(t_list)
    if t_max <= t_min:
        t_max = t_min + bin_s

    bins: Dict[int, Dict[str, Tuple[float, float]]] = {}

    for r in rows:
        node = r["node_id"]
        try:
            t = float(r["created_at"])
        except Exception:
            continue

        try:
            offset_ms = float(r["offset"]) * 1000.0
        except Exception:
            continue

        idx = int((t - t_min) / bin_s)
        if idx < 0:
            continue

        bucket = bins.setdefault(idx, {})
        prev = bucket.get(node)
        if (prev is None) or (t >= prev[0]):
            bucket[node] = (t, offset_ms)

    for idx, bucket in bins.items():
        if len(bucket) < 2:
            continue
        mean = sum(off for (_t, off) in bucket.values()) / len(bucket)
        for node, (t_last, off_ms) in list(bucket.items()):
            bucket[node] = (t_last, off_ms - mean)

    series: Dict[str, List[Dict[str, float]]] = {}
    for idx in sorted(bins.keys()):
        bucket = bins[idx]
        t_bin = t_min + (idx + 0.5) * bin_s
        for node, (_t, offset_ms) in bucket.items():
            series.setdefault(node, []).append({"t_wall": t_bin, "offset_ms": offset_ms})

    for node, pts in series.items():
        pts.sort(key=lambda p: p["t_wall"])
        prev = None
        for p in pts:
            if prev is None:
                p["delta_offset_ms"] = 0.0
            else:
                p["delta_offset_ms"] = p["offset_ms"] - prev
            prev = p["offset_ms"]

    def norm_pair(a: str, b: str) -> str:
        return "-".join(sorted([a, b]))

    pairs: Dict[str, List[Dict[str, float]]] = {}
    for idx in sorted(bins.keys()):
        bucket = bins[idx]
        nodes_now = sorted(bucket.keys())
        if len(nodes_now) < 2:
            continue

        t_bin = t_min + (idx + 0.5) * bin_s
        off = {n: bucket[n][1] for n in nodes_now}

        for i in range(len(nodes_now)):
            for j in range(i + 1, len(nodes_now)):
                a = nodes_now[i]
                b = nodes_now[j]
                pair_id = norm_pair(a, b)
                delta_ms = off[a] - off[b]
                pairs.setdefault(pair_id, []).append({"t_wall": t_bin, "delta_ms": delta_ms, "bin": float(idx)})

    jitter: Dict[str, Dict[str, float]] = {}
    for pair_id, pts in pairs.items():
        if len(pts) < JITTER_MIN_SAMPLES:
            continue
        deltas = [float(p["delta_ms"]) for p in pts]
        iqr = robust_iqr_ms(deltas)
        mad = robust_mad_ms(deltas)
        sigma = 0.7413 * iqr
        jitter[pair_id] = {
            "sigma_ms": float(sigma),
            "iqr_ms": float(iqr),
            "mad_ms": float(mad),
            "n": float(len(deltas)),
        }

    heatmap_data: List[Dict[str, Any]] = []
    if pairs and bins:
        idx_min = min(bins.keys())
        idx_max = max(bins.keys())
        total_bins = max(1, idx_max - idx_min + 1)
        display_bins = min(HEATMAP_MAX_BINS, total_bins)

        if display_bins == total_bins:
            for pair_id, pts in pairs.items():
                for p in pts:
                    heatmap_data.append({
                        "pair": pair_id,
                        "t_bin": float(p["t_wall"]),
                        "value": abs(float(p["delta_ms"]))
                    })
            n_bins = total_bins
        else:
            accum: Dict[Tuple[str, int], List[float]] = {}

            for pair_id, pts in pairs.items():
                for p in pts:
                    idx = int(p.get("bin", idx_min))
                    idx = max(idx_min, min(idx_max, idx))

                    rel = (idx - idx_min) / max(1.0, float(total_bins))
                    d_idx = int(rel * display_bins)
                    if d_idx >= display_bins:
                        d_idx = display_bins - 1

                    accum.setdefault((pair_id, d_idx), []).append(abs(float(p["delta_ms"])))

            for (pair_id, d_idx), vals in accum.items():
                idx_center = idx_min + (d_idx + 0.5) * (total_bins / display_bins)
                t_center = t_min + (idx_center + 0.5) * bin_s
                heatmap_data.append({
                    "pair": pair_id,
                    "t_bin": float(t_center),
                    "value": sum(vals) / max(1, len(vals))
                })

            n_bins = display_bins
    else:
        n_bins = 0

    return {
        "series": series,
        "pairs": pairs,
        "jitter": jitter,
        "heatmap": {"data": heatmap_data, "n_bins": n_bins},
        "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"},
    }


# ------------------------------------------------------------
# Link-metrics timeseries (theta/rtt/sigma per link, binned on created_at)
# ------------------------------------------------------------

def get_link_timeseries(
    window_seconds: int = WINDOW_SECONDS_DEFAULT,
    max_points: int = LINK_MAX_POINTS_DEFAULT,
    bin_s: float = BIN_S_DEFAULT
) -> Dict[str, Any]:
    """
    Reads link metrics from ntp_reference rows that contain peer_id and theta_ms/rtt_ms/sigma_ms.
    Uses created_at as global time axis (sink clock).
    Returns:
      links: { "C->D": [{t_wall, theta_ms, rtt_ms, sigma_ms}, ...], ... }
      latest_sigma: { "C->D": sigma_ms_latest_in_window, ... }
      meta: ...
    """
    conn = get_conn()
    cols = _table_cols(conn, "ntp_reference")
    needed = {"peer_id", "theta_ms", "rtt_ms", "sigma_ms", "created_at", "node_id"}
    if not needed.issubset(cols):
        conn.close()
        return {
            "links": {},
            "latest_sigma": {},
            "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at", "note": "link columns missing"},
        }

    cur = conn.cursor()
    cutoff = time.time() - float(window_seconds)

    rows = cur.execute(
        """
        SELECT node_id, peer_id, theta_ms, rtt_ms, sigma_ms, created_at
        FROM ntp_reference
        WHERE created_at >= ?
          AND peer_id IS NOT NULL
          AND (theta_ms IS NOT NULL OR rtt_ms IS NOT NULL OR sigma_ms IS NOT NULL)
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (cutoff, int(max_points)),
    ).fetchall()
    conn.close()

    if not rows:
        return {
            "links": {},
            "latest_sigma": {},
            "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"},
        }

    t_list: List[float] = []
    for r in rows:
        try:
            t_list.append(float(r["created_at"]))
        except Exception:
            pass
    if not t_list:
        return {
            "links": {},
            "latest_sigma": {},
            "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"},
        }

    t_min = min(t_list)

    def link_id(node_id: str, peer_id: str) -> str:
        # directed – because theta is signed "peer - self" for that node
        return f"{node_id}->{peer_id}"

    # bins: idx -> link -> (created_at, theta_ms, rtt_ms, sigma_ms)
    bins: Dict[int, Dict[str, Tuple[float, Optional[float], Optional[float], Optional[float]]]] = {}

    for r in rows:
        try:
            t = float(r["created_at"])
        except Exception:
            continue

        nid = str(r["node_id"])
        pid = str(r["peer_id"]) if r["peer_id"] is not None else None
        if not pid:
            continue

        lid = link_id(nid, pid)

        theta = r["theta_ms"]
        rtt = r["rtt_ms"]
        sig = r["sigma_ms"]

        theta_ms = float(theta) if theta is not None else None
        rtt_ms = float(rtt) if rtt is not None else None
        sigma_ms = float(sig) if sig is not None else None

        idx = int((t - t_min) / bin_s)
        if idx < 0:
            continue

        bucket = bins.setdefault(idx, {})
        prev = bucket.get(lid)
        # keep latest in bin
        if (prev is None) or (t >= prev[0]):
            bucket[lid] = (t, theta_ms, rtt_ms, sigma_ms)

    links: Dict[str, List[Dict[str, Any]]] = {}
    latest_sigma: Dict[str, float] = {}

    for idx in sorted(bins.keys()):
        bucket = bins[idx]
        t_bin = t_min + (idx + 0.5) * bin_s
        for lid, (_t, theta_ms, rtt_ms, sigma_ms) in bucket.items():
            obj = {"t_wall": t_bin}
            if theta_ms is not None:
                obj["theta_ms"] = theta_ms
            if rtt_ms is not None:
                obj["rtt_ms"] = rtt_ms
            if sigma_ms is not None:
                obj["sigma_ms"] = sigma_ms

            links.setdefault(lid, []).append(obj)

            if sigma_ms is not None:
                latest_sigma[lid] = sigma_ms

    # sort per link
    for lid, pts in links.items():
        pts.sort(key=lambda p: p["t_wall"])

    return {
        "links": links,
        "latest_sigma": latest_sigma,
        "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"},
    }


# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/")
def index():
    topo = get_topology()
    return render_template_string(TEMPLATE, topo=topo, db_path=str(DB_PATH))


@app.route("/api/topology")
def api_topology():
    try:
        return jsonify(get_topology())
    except Exception as e:
        return _json_error(f"/api/topology failed: {e}", 500)


@app.route("/api/status")
def api_status():
    try:
        cfg = load_config()
        ref = "C" if "C" in cfg else "A"
        return jsonify(get_status_snapshot(reference_node=ref))
    except Exception as e:
        return _json_error(f"/api/status failed: {e}", 500)


@app.route("/api/ntp_timeseries")
def api_ntp_timeseries():
    try:
        cfg = load_config()
        sync_cfg = (cfg.get("sync", {}) or {})

        window_s = int(sync_cfg.get("ui_window_s", WINDOW_SECONDS_DEFAULT))
        bin_s = float(sync_cfg.get("ui_bin_s", BIN_S_DEFAULT))
        max_points = int(sync_cfg.get("ui_max_points", MAX_POINTS_DEFAULT))

        return jsonify(get_ntp_timeseries(window_seconds=window_s, max_points=max_points, bin_s=bin_s))
    except Exception as e:
        return _json_error(f"/api/ntp_timeseries failed: {e}", 500)


@app.route("/api/link_timeseries")
def api_link_timeseries():
    try:
        cfg = load_config()
        sync_cfg = (cfg.get("sync", {}) or {})

        window_s = int(sync_cfg.get("ui_window_s", WINDOW_SECONDS_DEFAULT))
        bin_s = float(sync_cfg.get("ui_bin_s", BIN_S_DEFAULT))
        max_points = int(sync_cfg.get("ui_link_max_points", LINK_MAX_POINTS_DEFAULT))

        return jsonify(get_link_timeseries(window_seconds=window_s, max_points=max_points, bin_s=bin_s))
    except Exception as e:
        return _json_error(f"/api/link_timeseries failed: {e}", 500)


@app.template_filter("datetime_utc")
def datetime_utc(ts):
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


# ------------------------------------------------------------
# Template (adds Link Metrics charts)
# ------------------------------------------------------------

TEMPLATE = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>MeshTime Monitor</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 1.5rem; background: #111; color: #eee; }
    h1, h2, h3 { margin-top: 0; }
    .grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 2fr); gap: 1.5rem; align-items: flex-start; }
    @media (max-width: 1100px) { .grid { grid-template-columns: minmax(0, 1fr); } }
    .card { background: #1b1b1b; border-radius: 12px; padding: 1rem 1.25rem; box-shadow: 0 0 0 1px rgba(255,255,255,0.04); }
    table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    th, td { padding: 0.35rem 0.5rem; text-align: left; vertical-align: top; }
    th { border-bottom: 1px solid rgba(255,255,255,0.15); font-weight: 600; }
    tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
    .pill { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .pill-ok   { background: rgba(46,204,113,0.12); color:#2ecc71; }
    .pill-warn { background: rgba(241,196,15,0.12); color:#f1c40f; }
    .pill-bad  { background: rgba(231,76,60,0.12);  color:#e74c3c; }
    .small { font-size: 0.8rem; opacity: 0.75; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    canvas { max-width: 100%; }
    #meshCanvas { width: 100%; height: 260px; background: #171717; border-radius: 10px; }
    .plots-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; }
    @media (max-width: 1200px) { .plots-grid { grid-template-columns: minmax(0, 1fr); } }
    .footer { margin-top: 1rem; font-size: 0.8rem; opacity: 0.6; }
    .subline { margin-top: -0.3rem; }
    .spacer { height: 1rem; }
  </style>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@1.3.0/dist/chartjs-chart-matrix.min.js"></script>
</head>
<body>
  <h1>MeshTime Monitor</h1>
  <p class="small subline">
    Datenquelle: <span class="mono">{{ db_path }}</span>
    &nbsp;·&nbsp; X-Achse ist <strong>created_at (Sink-Clock)</strong> → sauberes Binning im unrooted Mesh
  </p>

  <div class="grid">
    <div>
      <div class="card">
        <h2>Aktueller Status pro Node</h2>
        <table id="statusTable">
          <thead>
            <tr>
              <th>Node</th>
              <th>Last Seen (UTC)</th>
              <th>Δ vs Ref (ms)</th>
              <th>Offset</th>
              <th>Age</th>
              <th>Sender Wallclock (UTC)</th>
            </tr>
          </thead>
          <tbody>
            <tr><td colspan="6" class="small">lade…</td></tr>
          </tbody>
        </table>
        <p class="small" id="statusMeta" style="margin-bottom:0;"></p>
      </div>

      <div class="card" style="margin-top:1rem;">
        <h2>Mesh-Topologie</h2>
        <canvas id="meshCanvas"></canvas>
        <p class="small">
          Grün = Node hat Daten, grau = bisher keine. Dicker Rand = <strong>Root</strong> (falls vorhanden).
        </p>
      </div>
    </div>

    <div>
      <div class="card">
        <h2>Offset Plots (letzte 10 Minuten)</h2>
        <div class="plots-grid">
          <div>
            <h3 style="font-size:0.95rem;">1) Paarweise ΔOffset (ms)</h3>
            <canvas id="pairChart" height="160"></canvas>
          </div>
          <div>
            <h3 style="font-size:0.95rem;">2) Offset-Änderung pro Node (Δoffset)</h3>
            <canvas id="offsetDeltaChart" height="160"></canvas>
          </div>
          <div>
            <h3 style="font-size:0.95rem;">3) Jitter der ΔOffset (robust σ ≈ 0.741·IQR)</h3>
            <canvas id="jitterBarChart" height="160"></canvas>
          </div>
          <div>
            <h3 style="font-size:0.95rem;">4) ΔOffset Heatmap (|Δ| gebinnt)</h3>
            <canvas id="heatmapChart" height="160"></canvas>
          </div>
        </div>
      </div>

      <div class="spacer"></div>

      <div class="card">
        <h2>Link Metrics (letzte 10 Minuten)</h2>
        <p class="small" style="margin-top:-0.5rem;">
          Quelle: <span class="mono">ntp_reference(peer_id, theta_ms, rtt_ms, sigma_ms)</span> — Link-ID ist <span class="mono">Node-&gt;Peer</span>.
        </p>
        <div class="plots-grid">
          <div>
            <h3 style="font-size:0.95rem;">5) θ pro Link (ms)</h3>
            <canvas id="thetaChart" height="160"></canvas>
          </div>
          <div>
            <h3 style="font-size:0.95rem;">6) RTT pro Link (ms)</h3>
            <canvas id="rttChart" height="160"></canvas>
          </div>
          <div>
            <h3 style="font-size:0.95rem;">7) Link σ (latest in window)</h3>
            <canvas id="linkSigmaBarChart" height="160"></canvas>
          </div>
          <div>
            <h3 style="font-size:0.95rem;">(Info)</h3>
            <div class="small">
              θ ist signiert (peer - self). Wenn θ stabil aber ΔOffset nicht → Bias/Regler.<br>
              RTT hoch/spiky → Funk/CoAP/Queueing. σ hoch → Jitter/Asymmetrie.
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    Wenn Δ-Linien flach bleiben und robust σ klein ist: Konsens-Maschine läuft.
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
    let thetaChart, rttChart, linkSigmaBarChart;

    function makeLineChart(ctx, yLabel) {
      return new Chart(ctx, {
        type: 'line',
        data: { datasets: [] },
        options: {
          responsive: true,
          animation: false,
          scales: {
            x: { type: 'time', time: { unit: 'second' }, ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y: { ticks: { color: '#aaa', callback: (v) => (v.toFixed ? v.toFixed(1) : v) + ' ' + yLabel }, grid: { color: 'rgba(255,255,255,0.05)' } }
          },
          plugins: {
            legend: { labels: { color: '#eee' } },
            tooltip: { callbacks: { label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)} ${yLabel}` } }
          },
          elements: { line: { tension: 0.1 }, point: { radius: 0 } }
        }
      });
    }

    function makeBarChart(ctx, label, yLabel) {
      return new Chart(ctx, {
        type: 'bar',
        data: { labels: [], datasets: [{ label: label, data: [], backgroundColor: 'rgba(52,152,219,0.7)' }] },
        options: {
          responsive: true,
          animation: false,
          scales: {
            x: { ticks: { color: '#aaa' }, grid: { display: false } },
            y: { ticks: { color: '#aaa', callback: (v) => v + ' ' + yLabel }, grid: { color: 'rgba(255,255,255,0.05)' } }
          },
          plugins: {
            legend: { labels: { color: '#eee' } },
            tooltip: { callbacks: { label: (ctx) => `${ctx.label}: ${ctx.parsed.y.toFixed(2)} ${yLabel}` } }
          }
        }
      });
    }

    function makeHeatmapChart(ctx) {
      return new Chart(ctx, {
        type: 'matrix',
        data: { datasets: [{
          label: 'ΔOffset Heatmap',
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
            return `rgba(${r},${g},150,0.85)`;
          },
          width: (context) => {
            const area = context.chart.chartArea || {};
            const nBins = context.chart._nBins || 1;
            const w = (area.right - area.left) / nBins;
            return (Number.isFinite(w) && w > 0) ? w : 10;
          },
          height: (context) => {
            const area = context.chart.chartArea || {};
            const cats = context.chart._categories || ['pair'];
            const nCats = cats.length || 1;
            const h = (area.bottom - area.top) / nCats;
            return (Number.isFinite(h) && h > 0) ? h : 10;
          }
        } ]},
        options: {
          responsive: true,
          animation: false,
          scales: {
            x: { type: 'time', time: { unit: 'second' }, ticks: { color: '#aaa' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            y: {
              type: 'linear',
              ticks: {
                color: '#aaa',
                callback: function (value) {
                  const cats = this.chart._categories || [];
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
                  const cats = ctx.chart._categories || [];
                  const idx = Math.round(ctx.raw.y);
                  const pairId = cats[idx] || '?';
                  const v = (ctx.raw && typeof ctx.raw.v === 'number') ? ctx.raw.v : 0;
                  const t = new Date(ctx.raw.x);
                  return `${pairId} @ ${t.toLocaleTimeString()} : |Δ|=${v.toFixed(2)} ms`;
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
      jitterBarChart   = makeBarChart(document.getElementById('jitterBarChart').getContext('2d'), 'Jitter (robust σ)', 'ms');
      heatmapChart     = makeHeatmapChart(document.getElementById('heatmapChart').getContext('2d'));

      thetaChart       = makeLineChart(document.getElementById('thetaChart').getContext('2d'), 'ms');
      rttChart         = makeLineChart(document.getElementById('rttChart').getContext('2d'), 'ms');
      linkSigmaBarChart= makeBarChart(document.getElementById('linkSigmaBarChart').getContext('2d'), 'σ (latest)', 'ms');
    }

    function updatePairChart(chart, pairs) {
      const pairIds = Object.keys(pairs || {}).sort();
      chart.data.datasets = [];
      pairIds.forEach((pairId, idx) => {
        const baseColor   = colors[idx % colors.length];
        const strokeColor = baseColor.replace('0.5', '0.9');
        const points = (pairs[pairId] || []).map(p => ({ x: new Date(p.t_wall * 1000), y: p.delta_ms }));
        chart.data.datasets.push({
          label: pairId, data: points,
          borderColor: strokeColor, backgroundColor: baseColor,
          fill: false, pointRadius: 0, borderWidth: 1.5
        });
      });
      chart.update();
    }

    function updateOffsetDeltaChart(chart, series) {
      const nodeIds = Object.keys(series || {}).sort();
      chart.data.datasets = [];
      nodeIds.forEach((nodeId, idx) => {
        const baseColor   = colors[idx % colors.length];
        const strokeColor = baseColor.replace('0.5', '0.9');
        const points = (series[nodeId] || []).map(p => ({ x: new Date(p.t_wall * 1000), y: p.delta_offset_ms }));
        chart.data.datasets.push({
          label: `Node ${nodeId}`, data: points,
          borderColor: strokeColor, backgroundColor: baseColor,
          fill: false, pointRadius: 0, borderWidth: 1.5
        });
      });
      chart.update();
    }

    function updateJitterBarChart(chart, jitter) {
      const ids = Object.keys(jitter || {}).sort();
      const labels = [];
      const data = [];
      const meta = {};

      ids.forEach(id => {
        const obj = jitter[id] || {};
        const sigma = obj.sigma_ms;
        if (sigma !== null && sigma !== undefined && Number.isFinite(sigma)) {
          labels.push(id);
          data.push(sigma);
          meta[id] = obj;
        }
      });

      chart.data.labels = labels;
      chart.data.datasets[0].data = data;
      chart._jitterMeta = meta;
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

    function updateLinkLineChart(chart, links, field, labelSuffix) {
      const ids = Object.keys(links || {}).sort();
      chart.data.datasets = [];
      ids.forEach((lid, idx) => {
        const baseColor   = colors[idx % colors.length];
        const strokeColor = baseColor.replace('0.5', '0.9');
        const pts = (links[lid] || [])
          .filter(p => p[field] !== null && p[field] !== undefined && Number.isFinite(p[field]))
          .map(p => ({ x: new Date(p.t_wall * 1000), y: p[field] }));
        chart.data.datasets.push({
          label: `${lid}${labelSuffix ? ' ' + labelSuffix : ''}`,
          data: pts,
          borderColor: strokeColor, backgroundColor: baseColor,
          fill: false, pointRadius: 0, borderWidth: 1.5
        });
      });
      chart.update();
    }

    function updateLinkSigmaBarChart(chart, latestSigma) {
      const ids = Object.keys(latestSigma || {}).sort();
      const labels = [];
      const data = [];
      ids.forEach(lid => {
        const v = latestSigma[lid];
        if (v !== null && v !== undefined && Number.isFinite(v)) {
          labels.push(lid);
          data.push(v);
        }
      });
      chart.data.labels = labels;
      chart.data.datasets[0].data = data;
      chart.update();
    }

    async function fetchJson(url) {
      const resp = await fetch(url);
      const data = await resp.json();
      if (!resp.ok) throw new Error(`${url}: ${data.error || 'unknown error'}`);
      return data;
    }

    async function fetchNtpData() { return await fetchJson('/api/ntp_timeseries'); }
    async function fetchLinkData() { return await fetchJson('/api/link_timeseries'); }
    async function fetchTopology() { return await fetchJson('/api/topology'); }
    async function fetchStatus() { return await fetchJson('/api/status'); }

    function classifyDelta(deltaAbsMs) {
      if (deltaAbsMs < 5) return 'pill-ok';
      if (deltaAbsMs < 20) return 'pill-warn';
      return 'pill-bad';
    }

    function renderStatusTable(status) {
      const tbody = document.querySelector('#statusTable tbody');
      tbody.innerHTML = '';

      const rows = (status && status.rows) || [];
      const ref = status && status.reference_node ? status.reference_node : '?';
      document.getElementById('statusMeta').textContent = rows.length ? `Referenz für Δ: Node ${ref} (Δ aus Offsets)` : '';

      if (!rows.length) {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td colspan="6" class="small">Keine ntp_reference-Daten gefunden.</td>`;
        tbody.appendChild(tr);
        return;
      }

      rows.forEach(r => {
        const node = r.node_id;
        const lastSeenUTC = r.t_mesh_utc || 'n/a';
        const offsetMs = (r.offset_ms !== null && r.offset_ms !== undefined) ? r.offset_ms : null;
        const ageS = (r.age_s !== null && r.age_s !== undefined) ? r.age_s : null;
        const dRef = (r.delta_vs_ref_ms !== null && r.delta_vs_ref_ms !== undefined) ? r.delta_vs_ref_ms : null;
        const tWallUTC = r.t_wall_utc || 'n/a';

        const cls = (dRef === null) ? 'pill-warn' : classifyDelta(Math.abs(dRef));
        const tr = document.createElement('tr');

        tr.innerHTML = `
          <td>${node}</td>
          <td>${lastSeenUTC}</td>
          <td>${dRef === null ? '<span class="small">n/a</span>' : `<span class="pill ${cls}">${dRef.toFixed(2)} ms</span>`}</td>
          <td>${offsetMs === null ? '<span class="small">n/a</span>' : `<span class="pill ${classifyDelta(Math.abs(offsetMs))}">${offsetMs.toFixed(2)} ms</span>`}</td>
          <td>${ageS === null ? '<span class="small">n/a</span>' : `<span class="small">${ageS.toFixed(1)} s</span>`}</td>
          <td class="small">${tWallUTC}</td>
        `;
        tbody.appendChild(tr);
      });
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
        const [ntpData, linkData, topo, status] = await Promise.all([
          fetchNtpData(),
          fetchLinkData(),
          fetchTopology(),
          fetchStatus()
        ]);

        renderStatusTable(status);

        updatePairChart(pairChart, ntpData.pairs || {});
        updateOffsetDeltaChart(offsetDeltaChart, ntpData.series || {});
        updateJitterBarChart(jitterBarChart, ntpData.jitter || {});
        updateHeatmapChart(heatmapChart, ntpData.heatmap || {data: [], n_bins: 0});

        const links = linkData.links || {};
        updateLinkLineChart(thetaChart, links, "theta_ms", "");
        updateLinkLineChart(rttChart, links, "rtt_ms", "");
        updateLinkSigmaBarChart(linkSigmaBarChart, linkData.latest_sigma || {});

        const activeNodes = new Set(Object.keys(ntpData.series || {}));
        const meshCanvas  = document.getElementById('meshCanvas');
        meshCanvas.width  = meshCanvas.clientWidth;
        meshCanvas.height = meshCanvas.clientHeight;
        drawMesh(meshCanvas, topo, activeNodes);
      } catch (e) {
        console.error('refresh failed:', e);
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
