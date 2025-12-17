# -*- coding: utf-8 -*-
# web_ui3_split.py (ROBUST DROP-IN)
#
# Works with nodes.json shaped like:
# {
#   "sync": {...},
#   "A": {"ip": "...", ...},
#   "B": {"ip": "...", ...},
#   "C": {"ip": "...", ...}
# }
#
# Also robust if diag_kalman / diag_controller tables are missing:
# -> returns empty datasets instead of HTTP 500

from __future__ import annotations

import threading
_snapshot_lock = threading.Lock()


import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request

from pathlib import Path

# project root = /home/pi/mesh_time
ROOT_DIR = Path(__file__).resolve().parent
if (ROOT_DIR / "mesh_data.sqlite").exists() and (ROOT_DIR / "config" / "nodes.json").exists():
    pass
else:
    # if file sits in /home/pi/mesh_time/mesh/, jump one up
    if (ROOT_DIR.parent / "mesh_data.sqlite").exists() and (ROOT_DIR.parent / "config" / "nodes.json").exists():
        ROOT_DIR = ROOT_DIR.parent

DB_PATH  = ROOT_DIR / "mesh_data.sqlite"
CFG_PATH = ROOT_DIR / "config" / "nodes.json"


WINDOW_S_DEFAULT = 10 * 60.0
SNAPSHOT_TTL_S = 1.0
UI_MAX_POINTS_DEFAULT = 6000

app = Flask(__name__)


# ------------------------------------------------------------
# SQLite helpers
# ------------------------------------------------------------

def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (name,),
    ).fetchone()
    return row is not None


def _float_or_none(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


# ------------------------------------------------------------
# Config parsing (robust for your shape)
# ------------------------------------------------------------

def _hash_color(node_id: str) -> str:
    # deterministic pseudo-random color
    h = 2166136261
    for ch in node_id.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    r = 64 + (((h >> 16) & 0xFF) % 160)
    g = 64 + (((h >> 8) & 0xFF) % 160)
    b = 64 + ((h & 0xFF) % 160)
    return f"#{r:02x}{g:02x}{b:02x}"


def load_nodes_from_cfg() -> Dict[str, Dict[str, Any]]:
    """
    Returns {node_id: node_cfg}.

    Heuristic: a "node" is any top-level entry whose value is a dict containing "ip".
    This matches your nodes.json with top-level "sync" (ignored) + "A/B/C" (kept).
    """
    if not CFG_PATH.exists():
        return {}

    try:
        cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if not isinstance(cfg, dict):
        return {}

    nodes: Dict[str, Dict[str, Any]] = {}
    for k, v in cfg.items():
        if not isinstance(v, dict):
            continue
        if "ip" not in v:
            continue
        nodes[str(k)] = v
    return nodes


def load_node_colors() -> Dict[str, str]:
    nodes = load_nodes_from_cfg()
    colors: Dict[str, str] = {}
    for nid, entry in nodes.items():
        c = entry.get("color")
        if isinstance(c, str) and c.startswith("#") and len(c) == 7:
            colors[nid] = c
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
    has_diag_kalman: bool
    has_diag_controller: bool


_snapshot_cache: Dict[str, Any] = {"ts": 0.0, "window_s": None, "data": None}


def _get_window_s() -> float:
    ws = request.args.get("window_s")
    if ws is None:
        return WINDOW_S_DEFAULT
    try:
        v = float(ws)
        return max(30.0, min(v, 24 * 3600.0))
    except Exception:
        return WINDOW_S_DEFAULT


def load_snapshot(window_s: float) -> Snapshot:
    now = time.time()

    with _snapshot_lock:
        cached = _snapshot_cache["data"]
        if (
            cached is not None
            and _snapshot_cache["window_s"] == window_s
            and (now - _snapshot_cache["ts"] < SNAPSHOT_TTL_S)
        ):
            return cached

    t_min = now - window_s

    # nodes+colors always available from config (even if DB tables missing)
    cfg_nodes = load_nodes_from_cfg()
    colors = load_node_colors()
    for nid in cfg_nodes.keys():
        colors.setdefault(nid, _hash_color(nid))

    kalman_rows_by_node: Dict[str, List[sqlite3.Row]] = {}
    ctrl_rows_by_node: Dict[str, List[sqlite3.Row]] = {}

    has_k = False
    has_c = False

    # DB part: soft-fail if tables not present
    if DB_PATH.exists():
        with get_conn() as conn:
            has_k = table_exists(conn, "diag_kalman")
            has_c = table_exists(conn, "diag_controller")

            if has_k:
                max_points = UI_MAX_POINTS_DEFAULT

                k_rows = conn.execute(
                    """
                    SELECT created_at_s AS created_at, node_id,
                           x_offset_ms, x_drift_ppm,
                           p_offset_ms2, p_drift_ppm2,
                           innov_med_ms, innov_p95_ms,
                           nis_med, nis_p95,
                           r_eff_ms2
                    FROM diag_kalman
                    WHERE created_at_s >= ?
                    ORDER BY created_at_s DESC
                    LIMIT ?
                    """,
                    (t_min, max_points),
                ).fetchall()[::-1]  # reverse back to ascending

                for r in k_rows:
                    nid = str(r["node_id"])
                    kalman_rows_by_node.setdefault(nid, []).append(r)

            if has_c:
                max_points = UI_MAX_POINTS_DEFAULT

                c_rows = conn.execute(
                    """
                    SELECT created_at_s AS created_at, node_id,
                           delta_desired_ms, delta_applied_ms,
                           slew_clipped
                    FROM diag_controller
                    WHERE created_at_s >= ?
                    ORDER BY created_at_s DESC
                    LIMIT ?
                    """,
                    (t_min, max_points),
                ).fetchall()[::-1]

                for r in c_rows:
                    nid = str(r["node_id"])
                    ctrl_rows_by_node.setdefault(nid, []).append(r)

    # union nodes from config + db
    nodes = sorted(set(cfg_nodes.keys()) | set(kalman_rows_by_node.keys()) | set(ctrl_rows_by_node.keys()))
    for nid in nodes:
        colors.setdefault(nid, _hash_color(nid))

    # determine t_max
    t_max = now
    for nid in nodes:
        rows = (kalman_rows_by_node.get(nid) or ctrl_rows_by_node.get(nid) or [])
        if rows:
            tt = _float_or_none(rows[-1]["created_at"])
            if tt is not None:
                t_max = max(t_max, tt)

    snap = Snapshot(
        window_s=window_s,
        t_min=t_min,
        t_max=t_max,
        nodes=nodes,
        colors=colors,
        kalman_rows_by_node=kalman_rows_by_node,
        ctrl_rows_by_node=ctrl_rows_by_node,
        has_diag_kalman=has_k,
        has_diag_controller=has_c,
    )

    _snapshot_cache["ts"] = now
    _snapshot_cache["window_s"] = window_s
    _snapshot_cache["data"] = snap
    return snap


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
        "db_path": str(DB_PATH),
        "has_diag_kalman": snap.has_diag_kalman,
        "has_diag_controller": snap.has_diag_controller,
    })


# -----------------------------
# Chart 1: Kalman state (offset, drift)
# -----------------------------
@app.route("/api/chart/kalman/state")
def api_kalman_state():
    snap = load_snapshot(_get_window_s())
    out = []
    for nid in snap.nodes:
        rows = snap.kalman_rows_by_node.get(nid, [])
        t, x_offset, x_drift = [], [], []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            x_offset.append(_float_or_none(r["x_offset_ms"]))
            x_drift.append(_float_or_none(r["x_drift_ppm"]))
        out.append({"node": nid, "t": t, "x_offset_ms": x_offset, "x_drift_ppm": x_drift})
    return jsonify(out)


# -----------------------------
# Chart 2: Kalman covariance (P)
# -----------------------------
@app.route("/api/chart/kalman/cov")
def api_kalman_cov():
    snap = load_snapshot(_get_window_s())
    out = []
    for nid in snap.nodes:
        rows = snap.kalman_rows_by_node.get(nid, [])
        t, p_off, p_drift = [], [], []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            p_off.append(_float_or_none(r["p_offset_ms2"]))
            p_drift.append(_float_or_none(r["p_drift_ppm2"]))
        out.append({"node": nid, "t": t, "p_offset_ms2": p_off, "p_drift_ppm2": p_drift})
    return jsonify(out)


# -----------------------------
# Chart 3: Innovation (median, p95)
# -----------------------------
@app.route("/api/chart/kalman/innov")
def api_kalman_innov():
    snap = load_snapshot(_get_window_s())
    out = []
    for nid in snap.nodes:
        rows = snap.kalman_rows_by_node.get(nid, [])
        t, med, p95 = [], [], []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            med.append(_float_or_none(r["innov_med_ms"]))
            p95.append(_float_or_none(r["innov_p95_ms"]))
        out.append({"node": nid, "t": t, "innov_med_ms": med, "innov_p95_ms": p95})
    return jsonify(out)


# -----------------------------
# Chart 4: NIS (median, p95)
# -----------------------------
@app.route("/api/chart/kalman/nis")
def api_kalman_nis():
    snap = load_snapshot(_get_window_s())
    out = []
    for nid in snap.nodes:
        rows = snap.kalman_rows_by_node.get(nid, [])
        t, med, p95 = [], [], []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            med.append(_float_or_none(r["nis_med"]))
            p95.append(_float_or_none(r["nis_p95"]))
        out.append({"node": nid, "t": t, "nis_med": med, "nis_p95": p95})
    return jsonify(out)


# -----------------------------
# Chart 5: Effective R
# -----------------------------
@app.route("/api/chart/kalman/r")
def api_kalman_r():
    snap = load_snapshot(_get_window_s())
    out = []
    for nid in snap.nodes:
        rows = snap.kalman_rows_by_node.get(nid, [])
        t, r_eff = [], []
        for r in rows:
            tt = _float_or_none(r["created_at"])
            if tt is None:
                continue
            t.append(tt)
            r_eff.append(_float_or_none(r["r_eff_ms2"]))
        out.append({"node": nid, "t": t, "r_eff_ms2": r_eff})
    return jsonify(out)


# -----------------------------
# Chart 8: Controller slew (desired vs applied + clipped markers)
# -----------------------------
@app.route("/api/chart/controller/slew")
def api_controller_slew():
    snap = load_snapshot(_get_window_s())
    out = []
    for nid in snap.nodes:
        rows = snap.ctrl_rows_by_node.get(nid, [])
        t, desired, applied, clipped = [], [], [], []
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)
