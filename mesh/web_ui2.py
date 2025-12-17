# -*- coding: utf-8 -*-
# mesh/web_ui.py
# MeshTime Dashboard v2: "One glance" Diagnose (window=10min, no buttons)
#
# Fixes vs previous:
# - Chart.js "ggplot-minimal-ish": thin, semi-transparent lines, light grid, no blur on resize
# - Proper canvas sizing: remove CSS forcing canvas height; use container boxes + maintainAspectRatio=false
# - Time axis in charts: seconds relative to "now" (last 10min) instead of huge epoch numbers
# - Optional unit auto-detect for link RTT/θ/σ (ms vs us) to avoid "300ms at home" surprises
# - Topology layout: spring-ish (force) layout, not all nodes on one line/circle → edges visible

from __future__ import annotations

import json
import math
import random
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template_string

BASE_DIR = Path(__file__).resolve().parent.parent   # /home/pi/mesh_time
DB_PATH  = BASE_DIR / "mesh_data.sqlite"
CFG_PATH = BASE_DIR / "config" / "nodes.json"

WINDOW_S = 10 * 60.0
BUCKET_S = 1.0  # 1s wall-time buckets for alignment

app = Flask(__name__)

# -----------------------------
# Helpers
# -----------------------------

def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def load_config() -> Dict[str, Any]:
    if not CFG_PATH.exists():
        return {}
    with CFG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def now_s() -> float:
    return float(time.time())

def robust_sigma_ms(xs_ms: List[float]) -> float:
    # robust sigma ~ 0.7413 * IQR
    if len(xs_ms) < 4:
        return 0.0
    ys = sorted(xs_ms)

    def q(p: float) -> float:
        n = len(ys)
        pos = p * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ys[lo]
        return ys[lo] * (1.0 - (pos - lo)) + ys[hi] * (pos - lo)

    iqr = q(0.75) - q(0.25)
    return float(0.7413 * iqr)

def p95_abs_ms(xs_ms: List[float]) -> float:
    if not xs_ms:
        return 0.0
    ys = sorted(abs(x) for x in xs_ms)
    idx = int(0.95 * (len(ys) - 1))
    return float(ys[idx])

def median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    mid = n // 2
    if n % 2 == 1:
        return float(ys[mid])
    return float(0.5 * (ys[mid - 1] + ys[mid]))

def safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

# -----------------------------
# Data extraction
# -----------------------------

def get_root_id(cfg: Dict[str, Any]) -> str:
    # Root is explicit is_root in nodes.json; fallback to "C"; then first node.
    try:
        nodes = cfg.get("nodes", cfg)  # allow both shapes
        if isinstance(nodes, dict):
            for nid, entry in nodes.items():
                sync = (entry.get("sync", {}) or {})
                if bool(sync.get("is_root", False)):
                    return str(nid)
            if "C" in nodes:
                return "C"
            return next(iter(nodes.keys()))
        elif isinstance(nodes, list):
            for entry in nodes:
                sync = (entry.get("sync", {}) or {})
                if bool(sync.get("is_root", False)):
                    return str(entry.get("id"))
            ids = [str(e.get("id")) for e in nodes if e.get("id")]
            if "C" in ids:
                return "C"
            return ids[0] if ids else "C"
    except Exception:
        pass
    return "C"

def list_nodes_and_links(cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    # Returns nodes=[{id, ip, color, is_root}], links=[{source,target}]
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []
    if not cfg:
        return nodes, links

    data = cfg.get("nodes", cfg)
    if isinstance(data, dict):
        for nid, entry in data.items():
            sync = (entry.get("sync", {}) or {})
            nodes.append({
                "id": str(nid),
                "ip": entry.get("ip"),
                "color": entry.get("color") or "#3498db",
                "is_root": bool(sync.get("is_root", False)),
                "neighbors": entry.get("neighbors", []) or []
            })
        by_id = {n["id"]: n for n in nodes}
        for n in nodes:
            for neigh in (n.get("neighbors") or []):
                if str(neigh) in by_id:
                    links.append({"source": n["id"], "target": str(neigh)})
    elif isinstance(data, list):
        for entry in data:
            sync = (entry.get("sync", {}) or {})
            nodes.append({
                "id": str(entry.get("id")),
                "ip": entry.get("ip"),
                "color": entry.get("color") or "#3498db",
                "is_root": bool(sync.get("is_root", False)),
                "neighbors": entry.get("neighbors", []) or []
            })
        by_id = {n["id"]: n for n in nodes}
        for n in nodes:
            for neigh in (n.get("neighbors") or []):
                if str(neigh) in by_id:
                    links.append({"source": n["id"], "target": str(neigh)})
    return nodes, links

def fetch_mesh_clock_window(conn, t0: float) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT node_id, t_wall_s, t_mesh_s, offset_s, err_mesh_vs_wall_s
        FROM mesh_clock
        WHERE t_wall_s >= ?
        ORDER BY t_wall_s ASC
        """,
        (t0,)
    ).fetchall()

def bucketize_mesh_clock(rows: List[sqlite3.Row]) -> Dict[Tuple[str, int], Dict[str, float]]:
    # For each (node_id, bucket=int(t_wall_s)), take latest sample in that bucket.
    out: Dict[Tuple[str, int], Dict[str, float]] = {}
    for r in rows:
        nid = str(r["node_id"])
        tw  = float(r["t_wall_s"])
        b = int(tw // BUCKET_S)
        out[(nid, b)] = {
            "t_wall_s": tw,
            "t_mesh_s": float(r["t_mesh_s"]),
            "offset_s": safe_float(r["offset_s"], 0.0) or 0.0,
            "err_mesh_vs_wall_s": safe_float(r["err_mesh_vs_wall_s"], 0.0) or 0.0,
        }
    return out

def compute_delta_to_root_series(rows: List[sqlite3.Row], root_id: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Returns per-node series: [(t_wall_s, delta_ms)] where delta = t_mesh(node)-t_mesh(root) aligned by wall buckets.
    """
    by = bucketize_mesh_clock(rows)
    root_buckets = {b for (nid, b) in by.keys() if nid == root_id}
    nodes = sorted({nid for (nid, _) in by.keys()})
    series: Dict[str, List[Tuple[float, float]]] = {nid: [] for nid in nodes if nid != root_id}

    for b in sorted(root_buckets):
        root = by.get((root_id, b))
        if not root:
            continue
        t = root["t_wall_s"]
        tr = root["t_mesh_s"]
        for nid in nodes:
            if nid == root_id:
                continue
            rr = by.get((nid, b))
            if not rr:
                continue
            di_ms = (rr["t_mesh_s"] - tr) * 1000.0
            series[nid].append((t, float(di_ms)))

    return series

def fetch_diag_controller(conn, t0: float) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT node_id, created_at_s, dt_s, delta_desired_ms, delta_applied_ms,
               slew_clipped, max_slew_ms_s, eff_eta
        FROM diag_controller
        WHERE created_at_s >= ?
        ORDER BY created_at_s ASC
        """,
        (t0,)
    ).fetchall()

def fetch_diag_kalman(conn, t0: float) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT node_id, created_at_s, n_meas,
               innov_med_ms, innov_p95_ms,
               nis_med, nis_p95,
               r_eff_ms2
        FROM diag_kalman
        WHERE created_at_s >= ?
        ORDER BY created_at_s ASC
        """,
        (t0,)
    ).fetchall()

def fetch_obs_link(conn, t0: float) -> List[sqlite3.Row]:
    return conn.execute(
        """
        SELECT node_id, peer_id, created_at_s, theta_ms, rtt_ms, sigma_ms, accepted, weight, reject_reason
        FROM obs_link
        WHERE created_at_s >= ?
        ORDER BY created_at_s ASC
        """,
        (t0,)
    ).fetchall()

def freshness_seconds(conn, t0: float) -> Tuple[float, Dict[str, float]]:
    now = now_s()
    rows = conn.execute(
        """
        SELECT node_id, MAX(t_wall_s) AS last_wall
        FROM mesh_clock
        WHERE t_wall_s >= ?
        GROUP BY node_id
        """,
        (t0,)
    ).fetchall()
    per: Dict[str, float] = {}
    for r in rows:
        nid = str(r["node_id"])
        lastw = safe_float(r["last_wall"], None)
        if lastw is None:
            continue
        per[nid] = max(0.0, now - float(lastw))
    max_age = max(per.values()) if per else 1e9
    return float(max_age), per

def _maybe_us_to_ms(values: List[float]) -> Tuple[List[float], str, float]:
    """
    Heuristic: if "ms" columns actually contain microseconds, convert.
    Rule of thumb:
      - typical home RTT is ~1..50 ms.
      - if median > 2000 (i.e. 2000 ms) OR median > 200 (still huge) and values are mostly in thousands,
        it might be microseconds mislabeled as ms.
    We decide:
      if median(values) > 300.0 and median(values)/1000.0 < 300.0 -> treat as us, divide by 1000.
    Returns (converted_values, unit_label, factor_applied)
    """
    xs = [x for x in values if x is not None and not math.isnan(x)]
    if not xs:
        return values, "ms", 1.0
    m = median(xs)
    # If m is big but dividing by 1000 becomes plausible
    if m > 300.0 and (m / 1000.0) < 300.0:
        return [x / 1000.0 if x is not None else None for x in values], "ms (auto: us→ms)", 0.001
    return values, "ms", 1.0

# -----------------------------
# KPI computation + diagnosis
# -----------------------------

def compute_system_kpis(conn, cfg: Dict[str, Any]) -> Dict[str, Any]:
    t0 = now_s() - WINDOW_S
    root_id = get_root_id(cfg)

    mc = fetch_mesh_clock_window(conn, t0)
    dc = fetch_diag_controller(conn, t0)
    dk = fetch_diag_kalman(conn, t0)
    ol = fetch_obs_link(conn, t0)

    delta_series = compute_delta_to_root_series(mc, root_id)

    all_deltas: List[float] = []
    for _nid, pts in delta_series.items():
        all_deltas.extend([v for (_t, v) in pts])
    residual_sigma = robust_sigma_ms(all_deltas)
    worst_pair_p95 = p95_abs_ms(all_deltas)

    thr_ms = float(((cfg.get("sync", {}) or {}).get("ui_converge_sigma_ms", 2.0)))
    converged = (residual_sigma > 0.0 and residual_sigma < thr_ms)

    since_s = 0.0
    if all_deltas:
        by_bucket: Dict[int, List[float]] = {}
        for _nid, pts in delta_series.items():
            for t, v in pts:
                b = int(float(t) // BUCKET_S)
                by_bucket.setdefault(b, []).append(float(v))
        buckets = sorted(by_bucket.keys())
        last_good_bucket = None
        for b in reversed(buckets):
            xs = by_bucket.get(b, [])
            if len(xs) < 2:
                continue
            s = robust_sigma_ms(xs)
            if s < thr_ms:
                last_good_bucket = b if last_good_bucket is None else last_good_bucket
                continue
            else:
                if last_good_bucket is not None:
                    break
        if last_good_bucket is not None:
            since_s = max(0.0, now_s() - (last_good_bucket * BUCKET_S))
    else:
        converged = False

    max_age, per_age = freshness_seconds(conn, t0)

    innov_med_by_node: Dict[str, float] = {}
    pnis_med_by_node: Dict[str, float] = {}
    pnis_outlier_rate_by_node: Dict[str, float] = {}
    if dk:
        tmp_innov: Dict[str, List[float]] = {}
        tmp_pnis: Dict[str, List[float]] = {}
        tmp_out: Dict[str, List[float]] = {}
        for r in dk:
            nid = str(r["node_id"])
            innov_med = safe_float(r["innov_med_ms"], None)
            nis_med = safe_float(r["nis_med"], None)
            nis_p95 = safe_float(r["nis_p95"], None)
            if innov_med is not None:
                tmp_innov.setdefault(nid, []).append(float(innov_med))
            if nis_med is not None:
                tmp_pnis.setdefault(nid, []).append(float(nis_med))
            if nis_p95 is not None:
                tmp_out.setdefault(nid, []).append(1.0 if float(nis_p95) > 9.0 else 0.0)
        for nid, xs in tmp_innov.items():
            innov_med_by_node[nid] = median(xs)
        for nid, xs in tmp_pnis.items():
            pnis_med_by_node[nid] = median(xs)
        for nid, xs in tmp_out.items():
            pnis_outlier_rate_by_node[nid] = float(sum(xs) / max(1, len(xs)))

    clip_rate_by_node: Dict[str, float] = {}
    eta_med_by_node: Dict[str, float] = {}
    if dc:
        tmp_clip: Dict[str, int] = {}
        tmp_cnt: Dict[str, int] = {}
        tmp_eta: Dict[str, List[float]] = {}
        for r in dc:
            nid = str(r["node_id"])
            clip = r["slew_clipped"]
            tmp_clip[nid] = tmp_clip.get(nid, 0) + (1 if int(clip or 0) != 0 else 0)
            tmp_cnt[nid] = tmp_cnt.get(nid, 0) + 1
            eta = safe_float(r["eff_eta"], None)
            if eta is not None:
                tmp_eta.setdefault(nid, []).append(float(eta))
        for nid in tmp_cnt:
            clip_rate_by_node[nid] = 100.0 * float(tmp_clip.get(nid, 0)) / float(max(1, tmp_cnt[nid]))
        for nid, xs in tmp_eta.items():
            eta_med_by_node[nid] = median(xs)

    link_sigma_med: Dict[str, float] = {}
    link_outlier_rate: Dict[str, float] = {}
    if ol:
        sig: Dict[str, List[float]] = {}
        out: Dict[str, int] = {}
        cnt: Dict[str, int] = {}
        for r in ol:
            a = str(r["node_id"])
            b = str(r["peer_id"])
            k = f"{a}→{b}"
            s = safe_float(r["sigma_ms"], None)
            if s is not None:
                sig.setdefault(k, []).append(float(s))
            accepted = int(r["accepted"] or 0)
            w = safe_float(r["weight"], 1.0) or 1.0
            bad = (accepted == 0) or (w < 0.2)
            out[k] = out.get(k, 0) + (1 if bad else 0)
            cnt[k] = cnt.get(k, 0) + 1
        for k, xs in sig.items():
            link_sigma_med[k] = median(xs)
        for k in cnt:
            link_outlier_rate[k] = 100.0 * float(out.get(k, 0)) / float(max(1, cnt[k]))

    link_sigma_med_val = median(list(link_sigma_med.values())) if link_sigma_med else 0.0
    link_sigma_worst = max(link_sigma_med.values()) if link_sigma_med else 0.0

    worst_clip_node = max(clip_rate_by_node, key=lambda k: clip_rate_by_node[k]) if clip_rate_by_node else None
    worst_innov_node = max(innov_med_by_node, key=lambda k: abs(innov_med_by_node[k])) if innov_med_by_node else None
    worst_link = max(link_sigma_med, key=lambda k: link_sigma_med[k]) if link_sigma_med else None

    stale = (max_age > float(((cfg.get("sync", {}) or {}).get("ui_stale_s", 5.0))))
    green = converged and (not stale) and ((worst_clip_node is None) or (clip_rate_by_node.get(worst_clip_node, 0.0) < 10.0))
    if stale:
        level = "RED"
        sentence = f"Stale: no fresh samples (max age {max_age:.1f}s)"
    elif green:
        clip_worst = clip_rate_by_node.get(worst_clip_node, 0.0) if worst_clip_node else 0.0
        sentence = f"Converged: residual σ={residual_sigma:.2f}ms, clip={clip_worst:.0f}%, link σ_med={link_sigma_med_val:.2f}ms"
        level = "GREEN"
    else:
        clip_w = clip_rate_by_node.get(worst_clip_node, 0.0) if worst_clip_node else 0.0
        if clip_w >= 20.0 and worst_clip_node:
            level = "YELLOW"
            sentence = f"Not stable: Node {worst_clip_node} clipped {clip_w:.0f}% (servo saturated)"
        elif worst_link and link_sigma_med.get(worst_link, 0.0) > max(2.0, 1.5 * link_sigma_med_val):
            level = "YELLOW"
            sentence = f"Not stable: link {worst_link} σ spikes (σ_med {link_sigma_med[worst_link]:.2f}ms)"
        elif worst_innov_node:
            level = "YELLOW"
            sentence = f"Not stable: innovation high on {worst_innov_node} (med {innov_med_by_node[worst_innov_node]:.2f}ms)"
        else:
            level = "YELLOW"
            sentence = "Not stable: convergence not confirmed"

    return {
        "meta": {"window_s": WINDOW_S, "root_id": root_id, "now_s": now_s()},
        "traffic": {"level": level, "sentence": sentence, "converged": bool(converged), "since_s": float(since_s), "stale": bool(stale)},
        "kpis": {
            "residual_sigma_ms": float(residual_sigma),
            "worst_pair_p95_ms": float(worst_pair_p95),
            "convergence_time_s": float(since_s) if converged else 0.0,
            "freshness_max_age_s": float(max_age),
            "innovation_med_abs_ms_worst": float(abs(innov_med_by_node[worst_innov_node])) if worst_innov_node else 0.0,
            "innovation_med_by_node": innov_med_by_node,
            "pnis_med_by_node": pnis_med_by_node,
            "pnis_outlier_rate_by_node": pnis_outlier_rate_by_node,
            "clip_rate_by_node": clip_rate_by_node,
            "eta_med_by_node": eta_med_by_node,
            "link_sigma_med_ms": link_sigma_med,
            "link_sigma_med_overall_ms": float(link_sigma_med_val),
            "link_sigma_worst_ms": float(link_sigma_worst),
            "link_outlier_rate_pct": link_outlier_rate,
        },
        "freshness_by_node_s": per_age,
    }

# -----------------------------
# API endpoints
# -----------------------------

@app.get("/api/meta")
def api_meta():
    cfg = load_config()
    root = get_root_id(cfg)
    with get_conn() as conn:
        t0 = now_s() - WINDOW_S
        max_age, _ = freshness_seconds(conn, t0)
    return jsonify({
        "app": "MeshTime v2",
        "window_s": WINDOW_S,
        "root_id": root,
        "update_age_s": max_age,
        "t_mesh_scale": "epoch_s",
        "build": "v2-dashboard-ggmin",
        "ts": now_s(),
    })

@app.get("/api/summary")
def api_summary():
    cfg = load_config()
    with get_conn() as conn:
        out = compute_system_kpis(conn, cfg)
    return jsonify(out)

@app.get("/api/rel_to_root")
def api_rel_to_root():
    cfg = load_config()
    root = get_root_id(cfg)
    t0 = now_s() - WINDOW_S
    with get_conn() as conn:
        mc = fetch_mesh_clock_window(conn, t0)
    series = compute_delta_to_root_series(mc, root)

    hist: Dict[str, Dict[str, Any]] = {}
    stats: Dict[str, Dict[str, float]] = {}
    for nid, pts in series.items():
        xs = [v for (_t, v) in pts]
        if not xs:
            hist[nid] = {"bins": [], "counts": []}
            stats[nid] = {"mean": 0.0, "median": 0.0, "p95abs": 0.0, "sigma": 0.0}
            continue

        mean_v = float(sum(xs) / len(xs))
        med = median(xs)
        p95a = p95_abs_ms(xs)
        sig = robust_sigma_ms(xs)

        lo = min(xs); hi = max(xs)
        span = max(1e-6, hi - lo)
        lo -= 0.05 * span
        hi += 0.05 * span
        nb = 30
        step = (hi - lo) / nb
        counts = [0] * nb
        for v in xs:
            i = int((v - lo) / step)
            i = 0 if i < 0 else (nb - 1 if i >= nb else i)
            counts[i] += 1
        bins = [float(lo + (i + 0.5) * step) for i in range(nb)]

        hist[nid] = {"bins": bins, "counts": counts}
        stats[nid] = {"mean": mean_v, "median": med, "p95abs": p95a, "sigma": sig}

    return jsonify({
        "root_id": root,
        "series": {nid: {"t": [t for (t, _v) in pts], "y": [v for (_t, v) in pts]} for nid, pts in series.items()},
        "hist": hist,
        "stats": stats,
    })

@app.get("/api/innovation")
def api_innovation():
    t0 = now_s() - WINDOW_S
    with get_conn() as conn:
        dk = fetch_diag_kalman(conn, t0)

    by_node: Dict[str, Dict[str, Any]] = {}
    for r in dk:
        nid = str(r["node_id"])
        t = float(r["created_at_s"])
        innov = safe_float(r["innov_med_ms"], None)
        p95 = safe_float(r["innov_p95_ms"], None)
        if innov is None:
            continue
        by_node.setdefault(nid, {"t": [], "innov": [], "p95": []})
        by_node[nid]["t"].append(t)
        by_node[nid]["innov"].append(float(innov))
        by_node[nid]["p95"].append(float(p95) if p95 is not None else None)
    return jsonify(by_node)

@app.get("/api/pnis")
def api_pnis():
    t0 = now_s() - WINDOW_S
    with get_conn() as conn:
        dk = fetch_diag_kalman(conn, t0)

    by_node: Dict[str, Dict[str, Any]] = {}
    for r in dk:
        nid = str(r["node_id"])
        t = float(r["created_at_s"])
        nis = safe_float(r["nis_med"], None)
        nis95 = safe_float(r["nis_p95"], None)
        if nis is None:
            continue
        by_node.setdefault(nid, {"t": [], "pnis": [], "pnis95": []})
        by_node[nid]["t"].append(t)
        by_node[nid]["pnis"].append(float(nis))
        by_node[nid]["pnis95"].append(float(nis95) if nis95 is not None else None)

    out = {}
    for nid, s in by_node.items():
        xs = [x for x in s["pnis"] if x is not None]
        out[nid] = 100.0 * sum(1 for x in xs if x > 9.0) / max(1, len(xs))
    return jsonify({"series": by_node, "outlier_pct": out})

@app.get("/api/controller")
def api_controller():
    t0 = now_s() - WINDOW_S
    with get_conn() as conn:
        dc = fetch_diag_controller(conn, t0)

    by_node: Dict[str, Dict[str, Any]] = {}
    for r in dc:
        nid = str(r["node_id"])
        t = float(r["created_at_s"])
        dd = safe_float(r["delta_desired_ms"], 0.0) or 0.0
        da = safe_float(r["delta_applied_ms"], 0.0) or 0.0
        clip = 1 if int(r["slew_clipped"] or 0) != 0 else 0
        eta = safe_float(r["eff_eta"], None)
        by_node.setdefault(nid, {"t": [], "desired": [], "applied": [], "clip": [], "eta": []})
        by_node[nid]["t"].append(t)
        by_node[nid]["desired"].append(float(dd))
        by_node[nid]["applied"].append(float(da))
        by_node[nid]["clip"].append(int(clip))
        by_node[nid]["eta"].append(float(eta) if eta is not None else None)
    return jsonify(by_node)

@app.get("/api/link")
def api_link():
    t0 = now_s() - WINDOW_S
    with get_conn() as conn:
        ol = fetch_obs_link(conn, t0)

    # Build series per link
    by_link: Dict[str, Dict[str, Any]] = {}
    all_rtt: List[float] = []
    all_theta: List[float] = []
    all_sigma: List[float] = []

    for r in ol:
        a = str(r["node_id"])
        b = str(r["peer_id"])
        k = f"{a}→{b}"
        t = float(r["created_at_s"])
        th = safe_float(r["theta_ms"], None)
        rtt = safe_float(r["rtt_ms"], None)
        sig = safe_float(r["sigma_ms"], None)
        acc = int(r["accepted"] or 0)
        w = safe_float(r["weight"], 1.0) or 1.0
        bad = (acc == 0) or (w < 0.2)

        by_link.setdefault(k, {"t": [], "theta": [], "rtt": [], "sigma": [], "bad": []})
        by_link[k]["t"].append(t)
        by_link[k]["theta"].append(th)
        by_link[k]["rtt"].append(rtt)
        by_link[k]["sigma"].append(sig)
        by_link[k]["bad"].append(1 if bad else 0)

        if rtt is not None: all_rtt.append(float(rtt))
        if th is not None: all_theta.append(float(th))
        if sig is not None: all_sigma.append(float(sig))

    # Auto unit sanity: if values look like microseconds but labeled ms, convert
    rtt_conv, rtt_unit, _ = _maybe_us_to_ms(all_rtt)
    th_conv, th_unit, _ = _maybe_us_to_ms(all_theta)
    sg_conv, sg_unit, _ = _maybe_us_to_ms(all_sigma)

    # Apply conversion back into by_link (keeping Nones)
    # We recompute conversion factors using the same heuristic output.
    # If unit string contains "us→ms", divide all link values by 1000.
    def apply_if_needed(unit: str, field: str):
        if "us→ms" not in unit:
            return
        for _k, s in by_link.items():
            s[field] = [x / 1000.0 if x is not None else None for x in s[field]]

    apply_if_needed(rtt_unit, "rtt")
    apply_if_needed(th_unit, "theta")
    apply_if_needed(sg_unit, "sigma")

    latest: Dict[str, Optional[float]] = {}
    for k, s in by_link.items():
        sigs = [x for x in s["sigma"] if x is not None]
        latest[k] = float(sigs[-1]) if sigs else None

    return jsonify({
        "series": by_link,
        "latest_sigma_ms": latest,
        "units": {
            "rtt": rtt_unit,
            "theta": th_unit,
            "sigma": sg_unit,
        }
    })

@app.get("/api/topology")
def api_topology():
    cfg = load_config()
    nodes, links = list_nodes_and_links(cfg)
    root = get_root_id(cfg)
    return jsonify({"nodes": nodes, "links": links, "root_id": root})

# -----------------------------
# UI
# -----------------------------

HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>MeshTime v2 · One glance</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root { color-scheme: dark; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background:#0b0f14; color:#e6edf3; }
    .wrap { padding: 14px 16px 18px; max-width: 1400px; margin: 0 auto; }
    .header { display:flex; gap:12px; align-items:center; justify-content:space-between; padding: 10px 12px; border:1px solid #1f2a37; border-radius: 12px; background:#0e141b; }
    .title { font-weight: 700; letter-spacing: .2px; }
    .meta { color:#9fb0c0; font-size: 12px; white-space: nowrap; }
    .ampel { width: 12px; height: 12px; border-radius: 50%; display:inline-block; margin-right: 10px; }
    .diag { display:flex; align-items:center; gap:8px; font-size: 13px; color:#cfe3f7; }
    .grid { margin-top: 12px; display:grid; grid-template-columns: 1.15fr 1fr; gap: 12px; }
    .grid2 { margin-top: 12px; display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .grid3 { margin-top: 12px; display:grid; grid-template-columns: 1fr 1fr 0.85fr; gap: 12px; }
    .card { border:1px solid #1f2a37; border-radius: 12px; background:#0e141b; padding: 10px 12px; }
    .card h3 { margin:0 0 10px 0; font-size: 13px; color:#cfe3f7; letter-spacing: .2px; font-weight: 650; }
    .kpis { display:grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .kpi { border:1px solid #162130; border-radius: 10px; padding: 8px 10px; background:#0b1118; }
    .kpi .k { font-size: 11px; color:#9fb0c0; }
    .kpi .v { font-size: 16px; font-weight: 700; margin-top: 2px; }
    .kpi .s { font-size: 11px; color:#9fb0c0; margin-top: 2px; }

    /* IMPORTANT: no forced canvas scaling → avoid blur */
    canvas { display:block; width:100%; height:100%; }

    .chartBox { height: 240px; }
    .chartBox.small { height: 190px; }
    .chartBox.tiny  { height: 160px; }

    .twoPlot { display:grid; grid-template-columns: 1.2fr 0.8fr; gap: 10px; align-items: start; }
    .statsBox { border:1px solid #162130; border-radius:10px; background:#0b1118; padding: 8px 10px; }
    .statsBox pre { margin:0; font-size: 11px; color:#cfe3f7; white-space: pre-wrap; }

    .mini { font-size: 11px; color:#9fb0c0; margin-top: 6px; }

    #topoWrap { height: 360px; border-radius: 10px; background:#0b1118; border:1px solid #162130; overflow:hidden; }
    #topo { width: 100%; height: 100%; }
    @media (max-width: 1100px){
      .grid, .grid2, .grid3 { grid-template-columns: 1fr; }
      .twoPlot { grid-template-columns: 1fr; }
      #topoWrap { height: 300px; }
    }
  </style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="title" id="hdrTitle">MeshTime v2</div>
    <div class="meta" id="hdrMeta">window=10min · update age=… · root=… · build=…</div>
  </div>

  <div class="card" style="margin-top:12px;">
    <div class="diag"><span class="ampel" id="lamp"></span><span id="diagText">…</span></div>
  </div>

  <div class="card" style="margin-top:12px;">
    <h3>System Summary</h3>
    <div class="kpis" id="kpiRowA"></div>
    <div style="height:8px;"></div>
    <div class="kpis" id="kpiRowB"></div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>Relative to Root · ΔMeshTime (ms)</h3>
      <div class="twoPlot">
        <div class="chartBox"><canvas id="plotDelta"></canvas></div>
        <div class="statsBox"><pre id="deltaStats">…</pre></div>
      </div>
    </div>
    <div class="card">
      <h3>Δ Distribution (10min)</h3>
      <div class="chartBox small"><canvas id="plotHist"></canvas></div>
      <div class="mini" id="histMini"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h3>Innovation (ms)</h3>
      <div class="chartBox"><canvas id="plotInnov"></canvas></div>
    </div>
    <div class="card">
      <h3>pNIS (dimensionless) · Outlier rate</h3>
      <div class="chartBox"><canvas id="plotPNIS"></canvas></div>
      <div class="mini" id="pnisMini"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h3>Controller Effort · desired vs applied (ms)</h3>
      <div class="chartBox"><canvas id="plotCtrl"></canvas></div>
    </div>
    <div class="card">
      <h3>Saturation · clip / η</h3>
      <div class="chartBox"><canvas id="plotClip"></canvas></div>
      <div class="mini" id="clipMini"></div>
    </div>
  </div>

  <div class="grid3">
    <div class="card">
      <h3>Link Quality · θ</h3>
      <div class="chartBox"><canvas id="plotTheta"></canvas></div>
      <div class="mini" id="thetaUnit">…</div>
    </div>
    <div class="card">
      <h3>Link Quality · RTT / σ_link</h3>
      <div class="chartBox"><canvas id="plotRTT"></canvas></div>
      <div class="mini" id="rttUnit">…</div>
    </div>
    <div class="card">
      <h3>Topology</h3>
      <div id="topoWrap"><canvas id="topo"></canvas></div>
      <div class="mini" id="topoMini"></div>
    </div>
  </div>
</div>

<script>
const fmt = (x, d=2) => (x===null||x===undefined||Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const fmtInt = (x) => (x===null||x===undefined||Number.isNaN(x)) ? "—" : String(Math.round(Number(x)));

const THEME = {
  fg:  "#cfe3f7",
  fg2: "#9fb0c0",
  grid:"rgba(159,176,192,0.10)",
  axis:"rgba(159,176,192,0.25)",
};

const PALETTE = [
  "52,152,219",  // blue
  "241,196,15",  // yellow
  "46,204,113",  // green
  "231,76,60",   // red
  "155,89,182",  // purple
  "26,188,156",  // teal
  "230,126,34"   // orange
];

function withAlpha(rgb, a){ return `rgba(${rgb},${a})`; }

function lamp(level){
  const el = document.getElementById("lamp");
  if(level==="GREEN") el.style.background = "#2ecc71";
  else if(level==="YELLOW") el.style.background = "#f1c40f";
  else el.style.background = "#e74c3c";
  el.style.boxShadow = "0 0 10px rgba(255,255,255,0.06)";
}

function kpiBox(k, v, s=""){
  const d = document.createElement("div");
  d.className="kpi";
  d.innerHTML = `<div class="k">${k}</div><div class="v">${v}</div><div class="s">${s}</div>`;
  return d;
}

function mkLine(canvas, datasets, yTitle){
  // minimal styling defaults
  datasets.forEach((ds, i)=>{
    ds.borderWidth = 1;
    ds.pointRadius = 0;
    ds.fill = false;
    ds.tension = 0.15;
    ds.borderColor = ds.borderColor || withAlpha(PALETTE[i % PALETTE.length], 0.65);
  });

  return new Chart(canvas, {
    type:"line",
    data:{ datasets },
    options:{
      responsive:true,
      maintainAspectRatio:false,
      animation:false,
      parsing:false,
      normalized:true,
      devicePixelRatio: window.devicePixelRatio || 1,
      scales:{
        x:{
          type:"linear",
          ticks:{ color:THEME.fg2, maxTicksLimit: 6, callback:(v)=> `${Math.round(v)}s` },
          grid:{ color:THEME.grid, borderColor:THEME.axis }
        },
        y:{
          ticks:{ color:THEME.fg2, maxTicksLimit: 6 },
          grid:{ color:THEME.grid, borderColor:THEME.axis },
          title:{ display:true, text:yTitle, color:THEME.fg2, font:{ size: 11 } }
        }
      },
      plugins:{
        legend:{ labels:{ color:THEME.fg, boxWidth: 10, boxHeight: 2 } },
        tooltip:{ mode:"nearest", intersect:false }
      }
    }
  });
}

function mkBar(canvas, labels, datasets, yTitle){
  datasets.forEach((ds, i)=>{
    ds.borderWidth = 0;
    ds.backgroundColor = ds.backgroundColor || withAlpha(PALETTE[i % PALETTE.length], 0.35);
  });

  return new Chart(canvas, {
    type:"bar",
    data:{ labels, datasets },
    options:{
      responsive:true,
      maintainAspectRatio:false,
      animation:false,
      devicePixelRatio: window.devicePixelRatio || 1,
      scales:{
        x:{ ticks:{ color:THEME.fg2, maxTicksLimit: 7 }, grid:{ color:THEME.grid, borderColor:THEME.axis } },
        y:{ ticks:{ color:THEME.fg2, maxTicksLimit: 5 }, grid:{ color:THEME.grid, borderColor:THEME.axis },
            title:{ display:true, text:yTitle, color:THEME.fg2, font:{ size: 11 } } }
      },
      plugins:{ legend:{ labels:{ color:THEME.fg, boxWidth: 10, boxHeight: 2 } } }
    }
  });
}

let C_delta=null, C_hist=null, C_innov=null, C_pnis=null, C_ctrl=null, C_clip=null, C_theta=null, C_rtt=null;

// Convert epoch seconds to "seconds ago" axis (0..WINDOW_S)
function toAgoSec(tEpoch, nowEpoch){
  return (tEpoch - nowEpoch); // negative = past, 0 = now
}

async function pull(){
  const [meta, summary, rel, innov, pnis, ctrl, link, topo] = await Promise.all([
    fetch("/api/meta").then(r=>r.json()),
    fetch("/api/summary").then(r=>r.json()),
    fetch("/api/rel_to_root").then(r=>r.json()),
    fetch("/api/innovation").then(r=>r.json()),
    fetch("/api/pnis").then(r=>r.json()),
    fetch("/api/controller").then(r=>r.json()),
    fetch("/api/link").then(r=>r.json()),
    fetch("/api/topology").then(r=>r.json()),
  ]);

  const nowEpoch = meta.ts || (Date.now()/1000);

  // Header
  const age = meta.update_age_s;
  document.getElementById("hdrTitle").textContent = `MeshTime v2 · window=10min`;
  document.getElementById("hdrMeta").textContent =
    `root=${meta.root_id} · update age=${fmt(age,1)}s · t_mesh=${meta.t_mesh_scale} · ${meta.build}`;

  // Ampel + Diagnose
  lamp(summary.traffic.level);
  const since = summary.traffic.converged ? ` · since ${fmt(summary.traffic.since_s,0)}s` : "";
  document.getElementById("diagText").textContent = `${summary.traffic.level}: ${summary.traffic.sentence}${since}`;

  // KPI board
  const A = document.getElementById("kpiRowA");
  const B = document.getElementById("kpiRowB");
  A.innerHTML=""; B.innerHTML="";

  const k = summary.kpis;
  A.appendChild(kpiBox("Residual σ (ms)", fmt(k.residual_sigma_ms,2)));
  A.appendChild(kpiBox("Worst pair p95 |Δ| (ms)", fmt(k.worst_pair_p95_ms,2)));
  A.appendChild(kpiBox("Convergence time (s)", k.convergence_time_s>0 ? fmtInt(k.convergence_time_s) : "—"));
  A.appendChild(kpiBox("Freshness max age (s)", fmt(k.freshness_max_age_s,1)));

  B.appendChild(kpiBox("Innovation |med| worst (ms)", fmt(k.innovation_med_abs_ms_worst,2)));
  const pnisVals = Object.values(k.pnis_med_by_node || {});
  B.appendChild(kpiBox("Consistency pNIS med", pnisVals.length? fmt(pnisVals.sort((a,b)=>b-a)[0],2):"—"));
  const clipVals = k.clip_rate_by_node || {};
  const clipWorst = Object.keys(clipVals).length ? Math.max(...Object.values(clipVals)) : null;
  B.appendChild(kpiBox("Controller saturation (clip %)", clipWorst!==null? fmt(clipWorst,0):"—"));
  B.appendChild(kpiBox("Link σ_med (ms)", fmt(k.link_sigma_med_overall_ms,2), `worst ${fmt(k.link_sigma_worst_ms,2)}`));

  // Panel 2A: delta to root (x axis: seconds ago)
  const dsDelta = [];
  const statsLines = [];
  let pi = 0;
  for(const [nid, s] of Object.entries(rel.series)){
    const pts = s.t.map((t,i)=>({x: toAgoSec(t, nowEpoch), y:s.y[i]}));
    dsDelta.push({ label:nid, data:pts, borderColor: withAlpha(PALETTE[pi++ % PALETTE.length], 0.65) });
    const st = rel.stats[nid] || {};
    statsLines.push(`${nid}: mean=${fmt(st.mean,2)}  med=${fmt(st.median,2)}  p95|Δ|=${fmt(st.p95abs,2)}  σ=${fmt(st.sigma,2)}`);
  }
  document.getElementById("deltaStats").textContent = statsLines.length? statsLines.join("\n") : "no data";

  if(!C_delta){
    C_delta = mkLine(document.getElementById("plotDelta"), dsDelta, "Δ to root (ms)");
  }else{
    C_delta.data.datasets = dsDelta;
    C_delta.update();
  }

  // Panel 2B: histogram
  const nodes = Object.keys(rel.hist || {});
  let labels = [];
  if(nodes.length){
    labels = (rel.hist[nodes[0]].bins || []).map(x=>fmt(x,1));
  }
  const dsH = [];
  nodes.forEach((nid, i)=>{
    const h = rel.hist[nid];
    dsH.push({ label:nid, data: h.counts || [], backgroundColor: withAlpha(PALETTE[i % PALETTE.length], 0.35) });
  });
  if(!C_hist){
    C_hist = mkBar(document.getElementById("plotHist"), labels, dsH, "count");
  }else{
    C_hist.data.labels = labels;
    C_hist.data.datasets = dsH;
    C_hist.update();
  }
  document.getElementById("histMini").textContent = nodes.length ? `Histogram bins ≈ ${nodes.length} nodes` : "no data";

  // Panel 3A: innovation (x axis seconds ago)
  const dsInnov = [];
  pi = 0;
  for(const [nid, s] of Object.entries(innov)){
    const pts = s.t.map((t,i)=>({x: toAgoSec(t, nowEpoch), y:s.innov[i]}));
    dsInnov.push({ label:nid, data:pts, borderColor: withAlpha(PALETTE[pi++ % PALETTE.length], 0.65) });
  }
  if(!C_innov){
    C_innov = mkLine(document.getElementById("plotInnov"), dsInnov, "innov (ms)");
  }else{
    C_innov.data.datasets = dsInnov;
    C_innov.update();
  }

  // Panel 3B: pNIS
  const dsPN = [];
  pi = 0;
  for(const [nid, s] of Object.entries(pnis.series || {})){
    const pts = s.t.map((t,i)=>({x: toAgoSec(t, nowEpoch), y:s.pnis[i]}));
    dsPN.push({ label:nid, data:pts, borderColor: withAlpha(PALETTE[pi++ % PALETTE.length], 0.65) });
  }
  if(!C_pnis){
    C_pnis = mkLine(document.getElementById("plotPNIS"), dsPN, "pNIS");
  }else{
    C_pnis.data.datasets = dsPN;
    C_pnis.update();
  }
  const out = pnis.outlier_pct || {};
  const outLines = Object.keys(out).length ? Object.entries(out).map(([k,v])=>`${k}: pNIS>9 = ${fmt(v,1)}%`).join(" · ") : "—";
  document.getElementById("pnisMini").textContent = outLines;

  // Panel 4A: controller desired vs applied
  const dsCtrl = [];
  pi = 0;
  for(const [nid, s] of Object.entries(ctrl)){
    const col = PALETTE[pi++ % PALETTE.length];
    const ptsD = s.t.map((t,i)=>({x: toAgoSec(t, nowEpoch), y:s.desired[i]}));
    const ptsA = s.t.map((t,i)=>({x: toAgoSec(t, nowEpoch), y:s.applied[i]}));
    dsCtrl.push({ label:`${nid} desired`, data:ptsD, borderColor: withAlpha(col, 0.35) });
    dsCtrl.push({ label:`${nid} applied`, data:ptsA, borderColor: withAlpha(col, 0.80) });
  }
  if(!C_ctrl){
    C_ctrl = mkLine(document.getElementById("plotCtrl"), dsCtrl, "delta (ms)");
  }else{
    C_ctrl.data.datasets = dsCtrl;
    C_ctrl.update();
  }

  // Panel 4B: clip & eta
  const dsClip = [];
  const clipLines = [];
  pi = 0;
  for(const [nid, s] of Object.entries(ctrl)){
    const col = PALETTE[pi++ % PALETTE.length];
    const ptsC = s.t.map((t,i)=>({x: toAgoSec(t, nowEpoch), y:s.clip[i]}));
    const etaPts = s.t.map((t,i)=>({x: toAgoSec(t, nowEpoch), y:(s.eta[i]===null? null : s.eta[i])})).filter(p=>p.y!==null);
    dsClip.push({ label:`${nid} clip`, data:ptsC, borderColor: withAlpha(col, 0.35) });
    dsClip.push({ label:`${nid} η`, data:etaPts, borderColor: withAlpha(col, 0.80) });

    const clipRate = 100*s.clip.reduce((a,b)=>a+b,0)/Math.max(1,s.clip.length);
    const etaVals = (s.eta||[]).filter(x=>x!==null);
    etaVals.sort((a,b)=>a-b);
    const etaMed = etaVals.length ? etaVals[Math.floor(etaVals.length/2)] : null;
    clipLines.push(`${nid}: clip=${fmt(clipRate,0)}% · η_med=${etaMed===null?"—":fmt(etaMed,2)}`);
  }
  if(!C_clip){
    C_clip = mkLine(document.getElementById("plotClip"), dsClip, "clip (0/1) & η");
  }else{
    C_clip.data.datasets = dsClip;
    C_clip.update();
  }
  document.getElementById("clipMini").textContent = clipLines.length? clipLines.join(" · ") : "—";

  // Panel 5: link quality
  const ls = link.series || {};
  const dsTheta = [];
  const dsRTT = [];
  pi = 0;
  for(const [lk, s] of Object.entries(ls)){
    const col = PALETTE[pi++ % PALETTE.length];
    const th = s.t.map((t,i)=>({x: toAgoSec(t, nowEpoch), y:s.theta[i]})).filter(p=>p.y!==null);
    const rttPts = s.t.map((t,i)=>({x: toAgoSec(t, nowEpoch), y:s.rtt[i]})).filter(p=>p.y!==null);
    const sigPts = s.t.map((t,i)=>({x: toAgoSec(t, nowEpoch), y:s.sigma[i]})).filter(p=>p.y!==null);
    dsTheta.push({ label:lk, data:th, borderColor: withAlpha(col, 0.70) });
    dsRTT.push({ label:`${lk} RTT`, data:rttPts, borderColor: withAlpha(col, 0.70) });
    dsRTT.push({ label:`${lk} σ`, data:sigPts, borderColor: withAlpha(col, 0.35) });
  }
  if(!C_theta){
    C_theta = mkLine(document.getElementById("plotTheta"), dsTheta, "θ (" + ((link.units && link.units.theta) ? link.units.theta : "ms") + ")");
  }else{
    C_theta.options.scales.y.title.text = "θ (" + ((link.units && link.units.theta) ? link.units.theta : "ms") + ")";
    C_theta.data.datasets = dsTheta;
    C_theta.update();
  }
  if(!C_rtt){
    C_rtt = mkLine(document.getElementById("plotRTT"), dsRTT, "RTT/σ (" + ((link.units && link.units.rtt) ? link.units.rtt : "ms") + ")");
  }else{
    C_rtt.options.scales.y.title.text = "RTT/σ (" + ((link.units && link.units.rtt) ? link.units.rtt : "ms") + ")";
    C_rtt.data.datasets = dsRTT;
    C_rtt.update();
  }

  document.getElementById("thetaUnit").textContent = (link.units && link.units.theta) ? `units: ${link.units.theta}` : "";
  document.getElementById("rttUnit").textContent = (link.units && link.units.rtt) ? `units: RTT=${link.units.rtt}, σ=${(link.units.sigma||"ms")}` : "";

  drawTopo(topo, link.latest_sigma_ms || {});
}

// --- Topology: simple force layout (few nodes, cheap, stable) ---
function drawTopo(topo, latestSig){
  const canvas = document.getElementById("topo");
  const ctx = canvas.getContext("2d");

  const dpr = window.devicePixelRatio || 1;
  const Wcss = canvas.clientWidth, Hcss = canvas.clientHeight;
  const W = canvas.width = Math.max(1, Math.floor(Wcss * dpr));
  const H = canvas.height = Math.max(1, Math.floor(Hcss * dpr));
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,W,H);

  const nodes = topo.nodes || [];
  const links = topo.links || [];
  const root = topo.root_id;

  if(!nodes.length){
    document.getElementById("topoMini").textContent = "no topology";
    return;
  }

  // init positions
  const pos = {};
  nodes.forEach((nd, i)=>{
    const rx = 0.2 + 0.6 * Math.random();
    const ry = 0.2 + 0.6 * Math.random();
    pos[nd.id] = { x: rx*W, y: ry*H, vx:0, vy:0, is_root: (nd.id===root), color: nd.color || "#3498db" };
  });

  // force sim (small number of iterations each refresh)
  const idSet = new Set(nodes.map(n=>n.id));
  const E = links.filter(l=>idSet.has(l.source) && idSet.has(l.target));
  const charge = -180.0 * dpr;
  const linkLen = 110.0 * dpr;
  const linkK = 0.006;
  const damp = 0.85;

  for(let it=0; it<60; it++){
    // repulsion
    for(let i=0; i<nodes.length; i++){
      for(let j=i+1; j<nodes.length; j++){
        const a = pos[nodes[i].id], b = pos[nodes[j].id];
        const dx = a.x - b.x, dy = a.y - b.y;
        const r2 = dx*dx + dy*dy + 1e-6;
        const f = charge / r2;
        const fx = f*dx, fy = f*dy;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }
    }
    // springs
    for(const l of E){
      const a = pos[l.source], b = pos[l.target];
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.sqrt(dx*dx + dy*dy) + 1e-6;
      const err = dist - linkLen;
      const f = linkK * err;
      const fx = f * (dx / dist), fy = f * (dy / dist);
      a.vx += fx; a.vy += fy;
      b.vx -= fx; b.vy -= fy;
    }
    // integrate + bounds
    for(const nd of nodes){
      const p = pos[nd.id];
      p.vx *= damp; p.vy *= damp;
      p.x += p.vx; p.y += p.vy;
      p.x = Math.max(14*dpr, Math.min(W-14*dpr, p.x));
      p.y = Math.max(14*dpr, Math.min(H-14*dpr, p.y));
    }
  }

  // link thickness by sigma (latest)
  const sigVals = Object.values(latestSig).filter(x=>x!==null && x!==undefined);
  sigVals.sort((a,b)=>a-b);
  const sigMed = sigVals.length ? sigVals[Math.floor(sigVals.length/2)] : 1.0;

  ctx.lineCap = "round";
  for(const l of E){
    const a = pos[l.source], b = pos[l.target];
    const k = `${l.source}→${l.target}`;
    const s = latestSig[k];
    let w = 1.0;
    if(s!==null && s!==undefined){
      w = 1.0 + 2.0 * Math.min(3.0, (s / Math.max(0.5, sigMed)));
    }
    ctx.strokeStyle = "rgba(159,176,192,0.25)";
    ctx.lineWidth = w * dpr * 0.9;
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  }

  // nodes + labels
  for(const nd of nodes){
    const p = pos[nd.id];
    const r = (p.is_root ? 7.5 : 6.5) * dpr;
    ctx.fillStyle = p.is_root ? "rgba(46,204,113,0.9)" : "rgba(52,152,219,0.85)";
    ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 2*Math.PI); ctx.fill();

    ctx.fillStyle = "rgba(230,237,243,0.85)";
    ctx.font = `${11*dpr}px system-ui`;
    ctx.fillText(nd.id, p.x + 9*dpr, p.y + 4*dpr);
  }

  document.getElementById("topoMini").textContent = `root=${root} · nodes=${nodes.length} · links=${E.length}`;
}

pull().catch(err=>{ console.error(err); });
setInterval(()=>pull().catch(err=>console.error(err)), 2000);
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return render_template_string(HTML)

def main():
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH}")
    app.run(host="0.0.0.0", port=8000, debug=False)

if __name__ == "__main__":
    main()
