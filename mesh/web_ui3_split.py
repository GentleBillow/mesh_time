# -*- coding: utf-8 -*-
# web_ui3_split.py
#
# MeshTime Dashboard – Split APIs + Kalman internals
# Endpoints:
#   GET /                      -> renders templates/convergence_split.html
#   GET /api/meta              -> nodes + colors + window
#   GET /api/chart/kalman/state
#   GET /api/chart/kalman/cov
#   GET /api/chart/kalman/innov
#   GET /api/chart/kalman/nis
#   GET /api/chart/kalman/r
#   GET /api/chart/controller/slew
#
# Design goals:
# - each chart has its own API
# - BUT: one cached snapshot so SQLite isn't hammered

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request

# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "mesh_data.sqlite"
CFG_PATH = BASE_DIR / "config" / "nodes.json"

WINDOW_S_DEFAULT = 10 * 60.0
SNAPSHOT_TTL_S = 1.0

app = Flask(__name__)

# ------------------------------------------------------------
# SQLite helper
# ------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

# ------------------------------------------------------------
# Config / Colors
# ------------------------------------------------------------

def _hash_color(node_id: str) -> str:
    # deterministic pseudo-random color (hex) from node id
    h = 2166136261
    for ch in node_id.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    r = (h >> 16) & 0xFF
    g = (h >> 8) & 0xFF
    b = h & 0xFF
    # avoid too-dark colors
    r = 64 + (r % 160)
    g = 64 + (g % 160)
    b = 64 + (b % 160)
    return f"#{r:02x}{g:02x}{b:02x}"

def load_node_colors() -> Dict[str, str]:
    colors: Dict[str, str] = {}
    if CFG_PATH.exists():
        try:
            cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
            nodes = cfg.get("nodes", cfg)  # support both {"nodes":{...}} and { ... }
            if isinstance(nodes, dict):
                for nid, entry in nodes.items():
                    if isinstance(entry, dict):
                        c = entry.get("color")
                        if isinstance(c, str) and c.startswith("#") and len(c) == 7:
                            colors[nid] = c
        except Exception:
            pass
    return colors

# ------------------------------------------------------------
# Snapshot cache
# ------------------------------------------------------------

@dataclass
class Snapshot:
    window_s: float
    t_min: float
    t_max: float
    nodes: List[str]
    colors: Dict[str, str]
    kalman_rows_by_node: Dict[str, List[sqlite3.Row]]
    ctrl_rows_by_node: Dict[str, List[sqlite3.Row]]

_snapshot_cache: Dict[str, Any] = {"ts": 0.0, "data": None}

def _float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def load_snapshot(window_s: float) -> Snapshot:
    now = time.time()
    cached = _snapshot_cache["data"]
    if cached is not None and (now - _snapshot_cache["ts"] < SNAPSHOT_TTL_S) and cached.window_s == window_s:
        return cached

    t_min = now - window_s
    colors = load_node_colors()

    kalman_rows_by_node: Dict[str, List[sqlite3.Row]] = {}
    ctrl_rows_by_node: Dict[str, List[sqlite3.Row]] = {}

    with get_conn() as conn:
        # diag_kalman
        k_rows = conn.execute(
            """
            SELECT created_at, node_id,
                   x_offset_ms, x_drift_ppm,
                   p_offset_ms2, p_drift_ppm2,
                   innov_med_ms, innov_p95_ms,
                   nis_med, nis_p95,
                   r_eff_ms2
            FROM diag_kalman
            WHERE created_at >= ?
            ORDER BY created_at ASC
            """,
            (t_min,),
        ).fetchall()

        for r in k_rows:
            nid = r["node_id"]
            kalman_rows_by_node.setdefault(nid, []).append(r)

        # diag_controller
        c_rows = conn.execute(
            """
            SELECT created_at, node_id,
                   delta_desired_ms, delta_applied_ms,
                   slew_clipped
            FROM diag_controller
            WHERE created_at >= ?
            ORDER BY created_at ASC
            """,
            (t_min,),
        ).fetchall()

        for r in c_rows:
            nid = r["node_id"]
            ctrl_rows_by_node.setdefault(nid, []).append(r)

    nodes = sorted(set(kalman_rows_by_node.keys()) | set(ctrl_rows_by_node.keys()))
    for nid in nodes:
        if nid not in colors:
            colors[nid] = _hash_color(nid)

    # determine t_max from available data
    t_max = now
    # try last kalman timestamp if present
    for nid in nodes:
        rows = kalman_rows_by_node.get(nid) or ctrl_rows_by_node.get(nid) or []
        if rows:
            t_last = _float_or_none(rows[-1]["created_at"])
            if t_last is not None:
                t_max = max(t_max, t_last)

    snap = Snapshot(
        window_s=window_s,
        t_min=t_min,
        t_max=t_max,
        nodes=nodes,
        colors=colors,
        kalman_rows_by_node=kalman_rows_by_node,
        ctrl_rows_by_node=ctrl_rows_by_node,
    )

    _snapshot_cache["ts"] = now
    _snapshot_cache["data"] = snap
    return snap

def _get_window_s() -> float:
    # optional: allow ?window_s=600
    ws = request.args.get("window_s", None)
    if ws is None:
        return WINDOW_S_DEFAULT
    try:
        v = float(ws)
        return max(30.0, min(v, 24 * 3600.0))
    except Exception:
        return WINDOW_S_DEFAULT

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/")
def index():
    return render_template("convergence_split.html")

@app.route("/api/meta")
def api_meta():
    window_s = _get_window_s()
    snap = load_snapshot(window_s)
    return jsonify({
        "window_s": snap.window_s,
        "t_min": snap.t_min,
        "t_max": snap.t_max,
        "nodes": snap.nodes,
        "colors": snap.colors,
    })

# -----------------------------
# Chart 1: Kalman state (offset, drift)
# -----------------------------
@app.route("/api/chart/kalman/state")
def api_kalman_state():
    window_s = _get_window_s()
    snap = load_snapshot(window_s)

    out = []
    for nid in snap.nodes:
        rows = snap.kalman_rows_by_node.get(nid, [])
        t = []
        x_offset = []
        x_drift = []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            x_offset.append(_float_or_none(r["x_offset_ms"]))
            x_drift.append(_float_or_none(r["x_drift_ppm"]))
        out.append({
            "node": nid,
            "t": t,
            "x_offset_ms": x_offset,
            "x_drift_ppm": x_drift,
        })
    return jsonify(out)

# -----------------------------
# Chart 2: Kalman covariance (P)
# -----------------------------
@app.route("/api/chart/kalman/cov")
def api_kalman_cov():
    window_s = _get_window_s()
    snap = load_snapshot(window_s)

    out = []
    for nid in snap.nodes:
        rows = snap.kalman_rows_by_node.get(nid, [])
        t = []
        p_off = []
        p_drift = []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            p_off.append(_float_or_none(r["p_offset_ms2"]))
            p_drift.append(_float_or_none(r["p_drift_ppm2"]))
        out.append({
            "node": nid,
            "t": t,
            "p_offset_ms2": p_off,
            "p_drift_ppm2": p_drift,
        })
    return jsonify(out)

# -----------------------------
# Chart 3: Innovation (median, p95)
# -----------------------------
@app.route("/api/chart/kalman/innov")
def api_kalman_innov():
    window_s = _get_window_s()
    snap = load_snapshot(window_s)

    out = []
    for nid in snap.nodes:
        rows = snap.kalman_rows_by_node.get(nid, [])
        t = []
        med = []
        p95 = []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            med.append(_float_or_none(r["innov_med_ms"]))
            p95.append(_float_or_none(r["innov_p95_ms"]))
        out.append({
            "node": nid,
            "t": t,
            "innov_med_ms": med,
            "innov_p95_ms": p95,
        })
    return jsonify(out)

# -----------------------------
# Chart 4: NIS (median, p95)
# -----------------------------
@app.route("/api/chart/kalman/nis")
def api_kalman_nis():
    window_s = _get_window_s()
    snap = load_snapshot(window_s)

    out = []
    for nid in snap.nodes:
        rows = snap.kalman_rows_by_node.get(nid, [])
        t = []
        med = []
        p95 = []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            med.append(_float_or_none(r["nis_med"]))
            p95.append(_float_or_none(r["nis_p95"]))
        out.append({
            "node": nid,
            "t": t,
            "nis_med": med,
            "nis_p95": p95,
        })
    return jsonify(out)

# -----------------------------
# Chart 5: Effective R
# -----------------------------
@app.route("/api/chart/kalman/r")
def api_kalman_r():
    window_s = _get_window_s()
    snap = load_snapshot(window_s)

    out = []
    for nid in snap.nodes:
        rows = snap.kalman_rows_by_node.get(nid, [])
        t = []
        r_eff = []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            r_eff.append(_float_or_none(r["r_eff_ms2"]))
        out.append({
            "node": nid,
            "t": t,
            "r_eff_ms2": r_eff,
        })
    return jsonify(out)

# -----------------------------
# Chart 8: Controller slew (desired vs applied + clipped markers)
# -----------------------------
@app.route("/api/chart/controller/slew")
def api_controller_slew():
    window_s = _get_window_s()
    snap = load_snapshot(window_s)

    out = []
    for nid in snap.nodes:
        rows = snap.ctrl_rows_by_node.get(nid, [])
        t = []
        desired = []
        applied = []
        clipped = []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            desired.append(_float_or_none(r["delta_desired_ms"]))
            applied.append(_float_or_none(r["delta_applied_ms"]))
            try:
                clipped.append(int(r["slew_clipped"]) if r["slew_clipped"] is not None else 0)
            except Exception:
                clipped.append(0)

        out.append({
            "node": nid,
            "t": t,
            "delta_desired_ms": desired,
            "delta_applied_ms": applied,
            "slew_clipped": clipped,
        })
    return jsonify(out)

# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
