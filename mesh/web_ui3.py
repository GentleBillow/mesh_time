# -*- coding: utf-8 -*-
# web_ui3_split.py
#
# MeshTime Dashboard – SPLIT API VERSION
# Each chart has its own endpoint:
#   /api/status
#   /api/chart/convergence
#   /api/chart/histogram
#   /api/chart/rtt
#   /api/chart/asymmetry
#   /api/chart/jitter
#   /api/chart/stats
#
# Internally everything is derived from ONE cached snapshot
# to avoid hammering SQLite.

from __future__ import annotations

import math
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from flask import Flask, jsonify, render_template

# ------------------------------------------------------------
# Paths & constants
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DB_PATH  = BASE_DIR / "mesh_data.sqlite"

WINDOW_S_DEFAULT = 10 * 60.0
SNAPSHOT_TTL_S   = 1.0   # cache lifetime

app = Flask(__name__)

# ------------------------------------------------------------
# SQLite helper
# ------------------------------------------------------------

def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

# ------------------------------------------------------------
# Snapshot cache
# ------------------------------------------------------------

_snapshot_cache: Dict[str, Any] = {
    "ts": 0.0,
    "data": None,
}

def load_snapshot(window_s: float = WINDOW_S_DEFAULT) -> Dict[str, Any]:
    now = time.time()
    if _snapshot_cache["data"] and now - _snapshot_cache["ts"] < SNAPSHOT_TTL_S:
        return _snapshot_cache["data"]

    t_min = now - window_s

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT created_at, src, dst,
                   theta_ms, rtt_ms, sigma_ms
            FROM ntp_reference
            WHERE created_at >= ?
            ORDER BY created_at ASC
            """,
            (t_min,)
        ).fetchall()

    by_link: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for r in rows:
        key = (r["src"], r["dst"])
        d = by_link.setdefault(key, {
            "t": [],
            "theta": [],
            "rtt": [],
            "sigma": [],
        })
        d["t"].append(r["created_at"])
        d["theta"].append(r["theta_ms"])
        d["rtt"].append(r["rtt_ms"])
        d["sigma"].append(r["sigma_ms"])

    snapshot = {
        "rows": rows,
        "by_link": by_link,
        "window_s": window_s,
        "n_samples": len(rows),
    }

    _snapshot_cache["ts"] = now
    _snapshot_cache["data"] = snapshot
    return snapshot

# ------------------------------------------------------------
# Robust helpers
# ------------------------------------------------------------

def robust_sigma(xs: List[float]) -> float:
    if not xs:
        return 0.0
    med = sorted(xs)[len(xs)//2]
    abs_dev = sorted(abs(x - med) for x in xs)
    mad = abs_dev[len(abs_dev)//2]
    return 1.4826 * mad if mad > 0 else 0.0

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------

@app.route("/")
def index():
    return render_template("convergence_split.html")

# -------------------- STATUS --------------------

@app.route("/api/status")
def api_status():
    snap = load_snapshot()
    thetas = [r["theta_ms"] for r in snap["rows"]]

    max_abs = max((abs(x) for x in thetas), default=0.0)
    cur_dev = thetas[-1] if thetas else 0.0

    return jsonify({
        "samples": snap["n_samples"],
        "window_s": snap["window_s"],
        "max_abs_ms": max_abs,
        "current_dev_ms": cur_dev,
    })

# -------------------- CONVERGENCE --------------------

@app.route("/api/chart/convergence")
def api_convergence():
    snap = load_snapshot()
    data = []

    for (src, dst), d in snap["by_link"].items():
        data.append({
            "id": f"{src}->{dst}",
            "t": d["t"],
            "theta": d["theta"],
        })

    return jsonify(data)

# -------------------- HISTOGRAM --------------------

@app.route("/api/chart/histogram")
def api_histogram():
    snap = load_snapshot()
    xs = [r["theta_ms"] for r in snap["rows"]]

    return jsonify({
        "values": xs,
        "sigma_robust": robust_sigma(xs),
    })

# -------------------- RTT --------------------

@app.route("/api/chart/rtt")
def api_rtt():
    snap = load_snapshot()
    out = []

    for (src, dst), d in snap["by_link"].items():
        out.append({
            "id": f"{src}->{dst}",
            "t": d["t"],
            "rtt": d["rtt"],
        })

    return jsonify(out)

# -------------------- ASYMMETRY --------------------

@app.route("/api/chart/asymmetry")
def api_asymmetry():
    snap = load_snapshot()
    out = []

    for (src, dst), d in snap["by_link"].items():
        if len(d["theta"]) < 2:
            continue
        asym = [abs(x) for x in d["theta"]]
        out.append({
            "id": f"{src}->{dst}",
            "t": d["t"],
            "asym": asym,
        })

    return jsonify(out)

# -------------------- JITTER --------------------

@app.route("/api/chart/jitter")
def api_jitter():
    snap = load_snapshot()
    out = []

    for (src, dst), d in snap["by_link"].items():
        sig = robust_sigma(d["theta"])
        out.append({
            "id": f"{src}->{dst}",
            "sigma_ms": sig,
        })

    return jsonify(out)

# -------------------- STATS --------------------

@app.route("/api/chart/stats")
def api_stats():
    snap = load_snapshot()
    out = []

    for (src, dst), d in snap["by_link"].items():
        if not d["rtt"]:
            continue
        avg = sum(d["rtt"]) / len(d["rtt"])
        out.append({
            "id": f"{src}->{dst}",
            "avg_rtt_ms": avg,
        })

    return jsonify(out)

# ------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True) 
