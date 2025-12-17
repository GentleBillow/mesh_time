# -*- coding: utf-8 -*-
# web_ui3_full.py - Complete MeshTime Dashboard with Sync + Links + Controller + Kalman
# FIXES:
#  - Add /api/kalman/all to avoid 5x polling
#  - Keep existing endpoints for compatibility
#  - Avoid Flask reloader double-process by default
#  - Optional tiny in-memory caching to reduce repeated SQLite scans

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Flask, jsonify, render_template

# ============================================================================
# Configuration
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent
DB_PATH = ROOT_DIR / "mesh_data.sqlite"
CFG_PATH = ROOT_DIR / "config" / "nodes.json"

DEFAULT_WINDOW_S = 600.0
NODE_IDS = ["A", "B", "C"]
LINK_IDS = ["A-B", "A-C", "B-C"]

NODE_COLORS = {
    "A": {"line": "rgb(200, 50, 50)", "fill": "rgba(200, 50, 50, 0.3)"},
    "B": {"line": "rgb(50, 180, 50)", "fill": "rgba(50, 180, 50, 0.3)"},
    "C": {"line": "rgb(50, 100, 200)", "fill": "rgba(50, 100, 200, 0.3)"},
}

LINK_COLORS = {
    "A-B": {"line": "rgb(200, 180, 0)", "fill": "rgba(200, 180, 0, 0.3)"},
    "A-C": {"line": "rgb(180, 50, 180)", "fill": "rgba(180, 50, 180, 0.3)"},
    "B-C": {"line": "rgb(0, 160, 160)", "fill": "rgba(0, 160, 160, 0.3)"},
}

app = Flask(__name__, template_folder=str(ROOT_DIR / "templates"))

# ============================================================================
# Micro cache (optional but helps a lot with polling)
# ============================================================================

_CACHE: Dict[str, Dict[str, Any]] = {}
# _CACHE[key] = {"t": unix_ts, "ttl": seconds, "value": obj}

def _cache_get(key: str) -> Optional[Any]:
    c = _CACHE.get(key)
    if not c:
        return None
    if time.time() - c["t"] <= c["ttl"]:
        return c["value"]
    return None

def _cache_set(key: str, value: Any, ttl: float) -> Any:
    _CACHE[key] = {"t": time.time(), "ttl": float(ttl), "value": value}
    return value

def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if (n % 2 == 1) else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

# ============================================================================
# Category 1: Synchronization Quality
# ============================================================================

def load_sync_data(db_path: str, window_s: float) -> Dict[str, List[Tuple[float, float]]]:
    cutoff = time.time() - window_s
    node_data = {nid: [] for nid in NODE_IDS}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
            SELECT created_at_s, node_id, offset_s
            FROM mesh_clock
            WHERE created_at_s > ?
            ORDER BY created_at_s ASC
        """
        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()
        conn.close()

        for t, node_id, offset_s in rows:
            if node_id in node_data and offset_s is not None:
                node_data[node_id].append((float(t), float(offset_s) * 1000.0))  # s -> ms
    except Exception as e:
        print(f"[sync_data] Error: {e}")

    return node_data

def interpolate_sync_timeline(node_data: Dict[str, List[Tuple[float, float]]]) -> Tuple[List[float], Dict[str, List[Optional[float]]]]:
    all_timestamps = set()
    for data in node_data.values():
        all_timestamps.update([t for t, _ in data])

    if not all_timestamps:
        return [], {nid: [] for nid in NODE_IDS}

    timestamps = sorted(all_timestamps)

    # limit points to keep JSON + frontend light
    if len(timestamps) > 500:
        step = max(1, len(timestamps) // 500)
        timestamps = [timestamps[i] for i in range(0, len(timestamps), step)]

    interpolated: Dict[str, List[Optional[float]]] = {nid: [] for nid in NODE_IDS}

    for nid in NODE_IDS:
        data = node_data[nid]
        if not data:
            interpolated[nid] = [None] * len(timestamps)
            continue

        data_t = [t for t, _ in data]
        data_v = [v for _, v in data]

        result: List[Optional[float]] = []
        for t in timestamps:
            if t < data_t[0] or t > data_t[-1]:
                result.append(None)
                continue

            # linear interpolation
            placed = False
            for i in range(len(data_t) - 1):
                if data_t[i] <= t <= data_t[i + 1]:
                    dt = data_t[i + 1] - data_t[i]
                    if dt > 0:
                        alpha = (t - data_t[i]) / dt
                        v = data_v[i] + alpha * (data_v[i + 1] - data_v[i])
                        result.append(float(v))
                    else:
                        result.append(float(data_v[i]))
                    placed = True
                    break
            if not placed:
                result.append(None)

        interpolated[nid] = result

    deviations: Dict[str, List[Optional[float]]] = {nid: [] for nid in NODE_IDS}
    for i in range(len(timestamps)):
        values = [interpolated[nid][i] for nid in NODE_IDS if interpolated[nid][i] is not None]
        if len(values) >= 2:
            med = _median([float(v) for v in values if v is not None])
            for nid in NODE_IDS:
                v = interpolated[nid][i]
                deviations[nid].append((float(v) - float(med)) if (v is not None and med is not None) else None)
        else:
            for nid in NODE_IDS:
                deviations[nid].append(None)

    return timestamps, deviations

def compute_sync_histogram(deviations: Dict[str, List[Optional[float]]], n_bins: int = 50) -> Dict[str, Any]:
    all_values: List[float] = []
    for vals in deviations.values():
        all_values.extend([float(v) for v in vals if v is not None])

    if not all_values:
        return {"bin_edges": [0, 1], "counts": {nid: [0] for nid in NODE_IDS}}

    min_val = min(all_values)
    max_val = max(all_values)
    rng = max_val - min_val
    if rng <= 0:
        rng = 1.0
    min_val -= 0.1 * rng
    max_val += 0.1 * rng

    bin_edges = list(np.linspace(min_val, max_val, n_bins + 1))

    counts: Dict[str, List[int]] = {}
    for nid in NODE_IDS:
        vals = [float(v) for v in deviations[nid] if v is not None]
        if vals:
            hist, _ = np.histogram(vals, bins=bin_edges)
            counts[nid] = hist.tolist()
        else:
            counts[nid] = [0] * n_bins

    return {"bin_edges": bin_edges, "counts": counts}

def compute_sync_stats(deviations: Dict[str, List[Optional[float]]], node_data: Dict[str, List[Tuple[float, float]]]) -> Dict[str, Any]:
    all_values: List[float] = []
    for vals in deviations.values():
        all_values.extend([float(v) for v in vals if v is not None])

    stats: Dict[str, Any] = {
        "n_samples": len(all_values),
        "std_dev_ms": float(np.std(all_values)) if all_values else 0.0,
        "max_abs_ms": float(max([abs(v) for v in all_values])) if all_values else 0.0,
        "nodes": {},
    }

    for nid in NODE_IDS:
        vals = [float(v) for v in deviations[nid] if v is not None]
        stats["nodes"][nid] = {
            "current_ms": vals[-1] if vals else None,
            "std_ms": float(np.std(vals)) if vals else 0.0,
            "n_samples": len(node_data.get(nid, [])),
        }

    return stats

@app.route("/api/sync")
def api_sync():
    key = f"sync:{int(DEFAULT_WINDOW_S)}"
    cached = _cache_get(key)
    if cached is not None:
        return jsonify(cached)

    node_data = load_sync_data(str(DB_PATH), DEFAULT_WINDOW_S)
    timestamps, deviations = interpolate_sync_timeline(node_data)
    payload = {
        "timeseries": {"timestamps": timestamps, "deviations": deviations},
        "histogram": compute_sync_histogram(deviations),
        "stats": compute_sync_stats(deviations, node_data),
        "colors": NODE_COLORS,
    }
    return jsonify(_cache_set(key, payload, ttl=1.0))  # 1s cache

# ============================================================================
# Category 2: Link Quality
# ============================================================================

def load_link_data(db_path: str, window_s: float) -> Dict[str, List[Dict[str, Optional[float]]]]:
    cutoff = time.time() - window_s
    link_data: Dict[str, List[Dict[str, Optional[float]]]] = {lid: [] for lid in LINK_IDS}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
            SELECT created_at_s, node_id, peer_id, rtt_ms, theta_ms, sigma_ms,
                   t1_s, t2_s, t3_s, t4_s
            FROM obs_link
            WHERE created_at_s > ? AND accepted = 1
            ORDER BY created_at_s ASC
        """
        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()
        conn.close()

        for t, node_id, peer_id, rtt_ms, theta_ms, sigma_ms, t1, t2, t3, t4 in rows:
            link_pair = tuple(sorted([node_id, peer_id]))
            link_id = f"{link_pair[0]}-{link_pair[1]}"
            if link_id not in link_data:
                continue

            asymmetry_ms = None
            if all(x is not None for x in [t1, t2, t3, t4]):
                forward = (t3 - t2) * 1000.0
                backward = (t4 - t1) * 1000.0
                asymmetry_ms = forward - backward

            link_data[link_id].append(
                {
                    "t": float(t),
                    "rtt": float(rtt_ms) if rtt_ms is not None else None,
                    "asymmetry": float(asymmetry_ms) if asymmetry_ms is not None else None,
                    "sigma": float(sigma_ms) if sigma_ms is not None else None,
                }
            )
    except Exception as e:
        print(f"[link_data] Error: {e}")

    return link_data

def compute_link_jitter(link_data: Dict[str, List[Dict[str, Optional[float]]]], window_size: int = 20) -> Dict[str, Dict[str, List[float]]]:
    jitter: Dict[str, Dict[str, List[float]]] = {}

    for link_id, data in link_data.items():
        if not data:
            jitter[link_id] = {"timestamps": [], "jitter": []}
            continue

        # keep alignment by using indices where rtt exists
        rtt_vals = [d["rtt"] for d in data]
        ts_vals = [d["t"] for d in data]

        out_t: List[float] = []
        out_j: List[float] = []

        for i in range(len(data)):
            if i < window_size - 1:
                continue
            window = [r for r in rtt_vals[i - window_size + 1 : i + 1] if r is not None]
            if len(window) >= 5:
                out_t.append(float(ts_vals[i]))
                out_j.append(float(np.std(window)))

        jitter[link_id] = {"timestamps": out_t, "jitter": out_j}

    return jitter

def compute_link_stats(link_data: Dict[str, List[Dict[str, Optional[float]]]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}

    for link_id, data in link_data.items():
        if not data:
            stats[link_id] = {"avg_rtt": 0.0, "avg_asymmetry": 0.0, "n_samples": 0}
            continue

        rtt_vals = [d["rtt"] for d in data if d["rtt"] is not None]
        asym_vals = [d["asymmetry"] for d in data if d["asymmetry"] is not None]

        stats[link_id] = {
            "avg_rtt": float(np.mean(rtt_vals)) if rtt_vals else 0.0,
            "avg_asymmetry": float(np.mean([abs(a) for a in asym_vals])) if asym_vals else 0.0,
            "n_samples": float(len(data)),
        }

    return stats

@app.route("/api/links")
def api_links():
    key = f"links:{int(DEFAULT_WINDOW_S)}"
    cached = _cache_get(key)
    if cached is not None:
        return jsonify(cached)

    link_data = load_link_data(str(DB_PATH), DEFAULT_WINDOW_S)
    jitter = compute_link_jitter(link_data)
    stats = compute_link_stats(link_data)

    timeseries: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for link_id, data in link_data.items():
        timeseries[link_id] = {
            "timestamps": [d["t"] for d in data],
            "rtt": [d["rtt"] for d in data],
            "asymmetry": [d["asymmetry"] for d in data],
            "sigma": [d["sigma"] for d in data],
        }

    payload = {"timeseries": timeseries, "jitter": jitter, "stats": stats, "colors": LINK_COLORS}
    return jsonify(_cache_set(key, payload, ttl=2.5))  # 2.5s cache

# ============================================================================
# Category 3: Controller Diagnostics
# ============================================================================

def load_controller_data(db_path: str, window_s: float) -> Dict[str, List[Dict[str, Any]]]:
    cutoff = time.time() - window_s
    node_data: Dict[str, List[Dict[str, Any]]] = {nid: [] for nid in NODE_IDS}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
            SELECT created_at_s, node_id, delta_desired_ms, delta_applied_ms, slew_clipped
            FROM diag_controller
            WHERE created_at_s > ?
            ORDER BY created_at_s ASC
        """
        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()
        conn.close()

        for t, node_id, desired, applied, clipped in rows:
            if node_id not in node_data:
                continue
            node_data[node_id].append(
                {
                    "t": float(t),
                    "desired": float(desired) if desired is not None else None,
                    "applied": float(applied) if applied is not None else None,
                    "clipped": int(clipped) if clipped is not None else 0,
                }
            )
    except Exception as e:
        print(f"[controller_data] Error: {e}")

    return node_data

def compute_controller_stats(node_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        if not data:
            stats[nid] = {"clip_rate": 0.0, "n_samples": 0.0}
            continue
        clipped_count = sum(1 for d in data if d.get("clipped", 0) == 1)
        stats[nid] = {"clip_rate": float(clipped_count / len(data)), "n_samples": float(len(data))}
    return stats

@app.route("/api/controller")
def api_controller():
    key = f"controller:{int(DEFAULT_WINDOW_S)}"
    cached = _cache_get(key)
    if cached is not None:
        return jsonify(cached)

    node_data = load_controller_data(str(DB_PATH), DEFAULT_WINDOW_S)
    stats = compute_controller_stats(node_data)

    timeseries: Dict[str, Dict[str, List[Any]]] = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        timeseries[nid] = {
            "timestamps": [d["t"] for d in data],
            "desired": [d["desired"] for d in data],
            "applied": [d["applied"] for d in data],
            "clipped": [d["clipped"] for d in data],
        }

    payload = {"timeseries": timeseries, "stats": stats, "colors": NODE_COLORS}
    return jsonify(_cache_set(key, payload, ttl=1.0))  # 1s cache

# ============================================================================
# Category 4: Kalman Filter Internals
# ============================================================================

def load_kalman_diagnostics(db_path: str, window_s: float) -> Dict[str, List[Dict[str, Optional[float]]]]:
    cutoff = time.time() - window_s
    node_data: Dict[str, List[Dict[str, Optional[float]]]] = {nid: [] for nid in NODE_IDS}

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = """
            SELECT created_at_s, node_id,
                   x_offset_ms, x_drift_ppm,
                   p_offset_ms2, p_drift_ppm2,
                   innov_med_ms, innov_p95_ms,
                   nis_med, nis_p95,
                   r_eff_ms2
            FROM diag_kalman
            WHERE created_at_s > ?
            ORDER BY created_at_s ASC
        """
        cursor.execute(query, (cutoff,))
        rows = cursor.fetchall()
        conn.close()

        for row in rows:
            (
                t,
                node_id,
                x_offset,
                x_drift,
                p_offset,
                p_drift,
                innov_med,
                innov_p95,
                nis_med,
                nis_p95,
                r_eff,
            ) = row

            if node_id not in node_data:
                continue

            node_data[node_id].append(
                {
                    "t": float(t),
                    "x_offset_ms": float(x_offset) if x_offset is not None else None,
                    "x_drift_ppm": float(x_drift) if x_drift is not None else None,
                    "p_offset_ms2": float(p_offset) if p_offset is not None else None,
                    "p_drift_ppm2": float(p_drift) if p_drift is not None else None,
                    "innov_med_ms": float(innov_med) if innov_med is not None else None,
                    "innov_p95_ms": float(innov_p95) if innov_p95 is not None else None,
                    "nis_med": float(nis_med) if nis_med is not None else None,
                    "nis_p95": float(nis_p95) if nis_p95 is not None else None,
                    "r_eff_ms2": float(r_eff) if r_eff is not None else None,
                }
            )
    except Exception as e:
        print(f"[kalman_diagnostics] Error: {e}")

    return node_data

def _format_kalman(node_data: Dict[str, List[Dict[str, Optional[float]]]]) -> Dict[str, Dict[str, List[Optional[float]]]]:
    out: Dict[str, Dict[str, List[Optional[float]]]] = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        out[nid] = {
            "timestamps": [d["t"] for d in data],
            "x_offset_ms": [d["x_offset_ms"] for d in data],
            "x_drift_ppm": [d["x_drift_ppm"] for d in data],
            "p_offset_ms2": [d["p_offset_ms2"] for d in data],
            "p_drift_ppm2": [d["p_drift_ppm2"] for d in data],
            "innov_med_ms": [d["innov_med_ms"] for d in data],
            "innov_p95_ms": [d["innov_p95_ms"] for d in data],
            "nis_med": [d["nis_med"] for d in data],
            "nis_p95": [d["nis_p95"] for d in data],
            "r_eff_ms2": [d["r_eff_ms2"] for d in data],
        }
    return out

@app.route("/api/kalman/all")
def api_kalman_all():
    # Slow-moving: cache longer
    key = f"kalman_all:{int(DEFAULT_WINDOW_S)}"
    cached = _cache_get(key)
    if cached is not None:
        return jsonify(cached)

    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    payload = {"data": _format_kalman(node_data), "colors": NODE_COLORS}
    return jsonify(_cache_set(key, payload, ttl=5.0))

# --- Keep old endpoints for compatibility (they now reuse load + format) ---

@app.route("/api/kalman/state")
def api_kalman_state():
    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    result = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        result[nid] = {
            "timestamps": [d["t"] for d in data],
            "x_offset_ms": [d["x_offset_ms"] for d in data],
            "x_drift_ppm": [d["x_drift_ppm"] for d in data],
        }
    return jsonify({"data": result, "colors": NODE_COLORS})

@app.route("/api/kalman/covariance")
def api_kalman_covariance():
    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    result = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        result[nid] = {
            "timestamps": [d["t"] for d in data],
            "p_offset_ms2": [d["p_offset_ms2"] for d in data],
            "p_drift_ppm2": [d["p_drift_ppm2"] for d in data],
        }
    return jsonify({"data": result, "colors": NODE_COLORS})

@app.route("/api/kalman/innovation")
def api_kalman_innovation():
    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    result = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        result[nid] = {
            "timestamps": [d["t"] for d in data],
            "innov_med_ms": [d["innov_med_ms"] for d in data],
            "innov_p95_ms": [d["innov_p95_ms"] for d in data],
        }
    return jsonify({"data": result, "colors": NODE_COLORS})

@app.route("/api/kalman/nis")
def api_kalman_nis():
    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    result = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        result[nid] = {
            "timestamps": [d["t"] for d in data],
            "nis_med": [d["nis_med"] for d in data],
            "nis_p95": [d["nis_p95"] for d in data],
        }
    return jsonify({"data": result, "colors": NODE_COLORS})

@app.route("/api/kalman/r_eff")
def api_kalman_r_eff():
    node_data = load_kalman_diagnostics(str(DB_PATH), DEFAULT_WINDOW_S)
    result = {}
    for nid in NODE_IDS:
        data = node_data[nid]
        result[nid] = {
            "timestamps": [d["t"] for d in data],
            "r_eff_ms2": [d["r_eff_ms2"] for d in data],
        }
    return jsonify({"data": result, "colors": NODE_COLORS})

# ============================================================================
# Routes
# ============================================================================

@app.route("/")
def index():
    return render_template("convergence_full.html")

if __name__ == "__main__":
    # IMPORTANT: use_reloader=False prevents Flask from spawning 2 processes (double load / weirdness)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, use_reloader=False)
