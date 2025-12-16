# -*- coding: utf-8 -*-
# mesh/web_ui.py
# MeshTime Dashboard v2: "One glance" Diagnose (window=10min, no buttons)

from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
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

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# -----------------------------
# Data extraction
# -----------------------------

def get_root_id(cfg: Dict[str, Any]) -> str:
    # Root is explicit is_root in nodes.json; fallback to "C"; then first node.
    try:
        nodes = cfg.get("nodes", cfg)  # allow both shapes
        if isinstance(nodes, dict):
            # older style: top-level dict keyed by node_id
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
    nodes = []
    links = []
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
        # links
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
    # mesh_clock has created_at_s? you use t_wall_s, created_at_s sometimes.
    # We rely on t_wall_s as "time axis" in epoch seconds.
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
        key = (nid, b)
        # latest wins (since sorted asc, just overwrite)
        out[key] = {
            "t_wall_s": tw,
            "t_mesh_s": float(r["t_mesh_s"]),
            "offset_s": safe_float(r["offset_s"], 0.0) or 0.0,
            "err_mesh_vs_wall_s": safe_float(r["err_mesh_vs_wall_s"], 0.0) or 0.0,
        }
    return out

def compute_delta_to_root_series(rows: List[sqlite3.Row], root_id: str) -> Dict[str, List[Tuple[float, float]]]:
    """
    Returns per-node series: [(t_wall_s, delta_ms)] where delta = t_mesh(node)-t_mesh(root) aligned by wall buckets.
    Root itself omitted (implicitly 0).
    """
    by = bucketize_mesh_clock(rows)
    # find buckets present for root
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
    """
    Freshness = max age across nodes (based on latest mesh_clock per node).
    Returns (max_age, per_node_age).
    """
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
    per = {}
    for r in rows:
        nid = str(r["node_id"])
        lastw = safe_float(r["last_wall"], None)
        if lastw is None:
            continue
        per[nid] = max(0.0, now - float(lastw))
    max_age = max(per.values()) if per else 1e9
    return float(max_age), per

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

    # Outcome: residual sigma over all Δ samples (A,B,... relative to root)
    all_deltas = []
    for nid, pts in delta_series.items():
        all_deltas.extend([v for (_t, v) in pts])
    residual_sigma = robust_sigma_ms(all_deltas)
    worst_pair_p95 = p95_abs_ms(all_deltas)

    # Converged since when: "sigma < threshold" stable (simple heuristic)
    thr_ms = float(((cfg.get("sync", {}) or {}).get("ui_converge_sigma_ms", 2.0)))
    # compute rolling sigma over last N seconds using last ~120 samples
    converged = (residual_sigma > 0.0 and residual_sigma < thr_ms)

    # crude "since": look back from newest bucket until sigma breaks
    # (fast enough for small windows; avoids heavy SQL)
    since_s = 0.0
    if all_deltas:
        # rebuild a time-sorted list of (t, delta_ms across nodes) for sigma over time buckets
        # take per-bucket all node deltas present and compute sigma; walk backwards
        by_bucket: Dict[int, List[float]] = {}
        for nid, pts in delta_series.items():
            for t, v in pts:
                b = int(float(t) // BUCKET_S)
                by_bucket.setdefault(b, []).append(float(v))
        buckets = sorted(by_bucket.keys())
        last_good_bucket = None
        # walk backwards; require at least 5 samples to decide
        for b in reversed(buckets):
            xs = by_bucket.get(b, [])
            if len(xs) < 2:
                continue
            s = robust_sigma_ms(xs)
            if s < thr_ms:
                last_good_bucket = b if last_good_bucket is None else last_good_bucket
                continue
            else:
                # break at first violation after a streak
                if last_good_bucket is not None:
                    break
        if last_good_bucket is not None:
            since_s = max(0.0, now_s() - (last_good_bucket * BUCKET_S))
    else:
        converged = False

    # Freshness
    max_age, per_age = freshness_seconds(conn, t0)

    # Mechanic: innovation + pNIS + saturation
    innov_med_by_node = {}
    pnis_med_by_node = {}
    pnis_outlier_rate_by_node = {}  # pNIS > 9 share
    if dk:
        tmp_innov = {}
        tmp_pnis = {}
        tmp_out = {}
        tmp_n = {}
        for r in dk:
            nid = str(r["node_id"])
            innov_med = safe_float(r["innov_med_ms"], None)
            nis_med = safe_float(r["nis_med"], None)
            nis_p95 = safe_float(r["nis_p95"], None)
            if innov_med is not None:
                tmp_innov.setdefault(nid, []).append(float(innov_med))
            if nis_med is not None:
                tmp_pnis.setdefault(nid, []).append(float(nis_med))
            # outlier proxy: nis_p95 > 9 counts as "often bad"
            if nis_p95 is not None:
                tmp_out.setdefault(nid, []).append(1.0 if float(nis_p95) > 9.0 else 0.0)
            tmp_n[nid] = tmp_n.get(nid, 0) + 1
        for nid, xs in tmp_innov.items():
            innov_med_by_node[nid] = median(xs)
        for nid, xs in tmp_pnis.items():
            pnis_med_by_node[nid] = median(xs)
        for nid, xs in tmp_out.items():
            pnis_outlier_rate_by_node[nid] = float(sum(xs) / max(1, len(xs)))

    # Saturation: clip_rate per node from diag_controller
    clip_rate_by_node = {}
    eta_med_by_node = {}
    if dc:
        tmp_clip = {}
        tmp_cnt = {}
        tmp_eta = {}
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

    # Link noise: median sigma_ms per link; outlier rate via accepted/weight
    link_sigma_med = {}
    link_outlier_rate = {}
    if ol:
        sig = {}
        out = {}
        cnt = {}
        for r in ol:
            a = str(r["node_id"])
            b = str(r["peer_id"])
            k = f"{a}→{b}"
            s = safe_float(r["sigma_ms"], None)
            if s is not None:
                sig.setdefault(k, []).append(float(s))
            accepted = int(r["accepted"] or 0)
            w = safe_float(r["weight"], 1.0) or 1.0
            # outlier proxy: not accepted OR very low weight
            bad = (accepted == 0) or (w < 0.2)
            out[k] = out.get(k, 0) + (1 if bad else 0)
            cnt[k] = cnt.get(k, 0) + 1
        for k, xs in sig.items():
            link_sigma_med[k] = median(xs)
        for k in cnt:
            link_outlier_rate[k] = 100.0 * float(out.get(k, 0)) / float(max(1, cnt[k]))

    link_sigma_med_val = median(list(link_sigma_med.values())) if link_sigma_med else 0.0
    link_sigma_worst = max(link_sigma_med.values()) if link_sigma_med else 0.0

    # Worst offenders
    worst_clip_node = max(clip_rate_by_node, key=lambda k: clip_rate_by_node[k]) if clip_rate_by_node else None
    worst_innov_node = max(innov_med_by_node, key=lambda k: abs(innov_med_by_node[k])) if innov_med_by_node else None
    worst_link = max(link_sigma_med, key=lambda k: link_sigma_med[k]) if link_sigma_med else None

    # Diagnosis sentence + traffic light
    # Stale if max_age > 5s (tunable)
    stale = (max_age > float(((cfg.get("sync", {}) or {}).get("ui_stale_s", 5.0))))
    green = converged and not stale and ( (worst_clip_node is None) or (clip_rate_by_node.get(worst_clip_node, 0.0) < 10.0) )
    if stale:
        level = "RED"
        sentence = f"Stale: no fresh samples (max age {max_age:.1f}s)"
    elif green:
        clip_worst = clip_rate_by_node.get(worst_clip_node, 0.0) if worst_clip_node else 0.0
        sentence = f"Converged: residual σ={residual_sigma:.2f}ms, clip={clip_worst:.0f}%, link σ_med={link_sigma_med_val:.2f}ms"
        level = "GREEN"
    else:
        # yellow diagnosis: pick main culprit
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
        "meta": {
            "window_s": WINDOW_S,
            "root_id": root_id,
            "now_s": now_s(),
        },
        "traffic": {
            "level": level,
            "sentence": sentence,
            "converged": bool(converged),
            "since_s": float(since_s),
            "stale": bool(stale),
        },
        "kpis": {
            # Row A – Outcome
            "residual_sigma_ms": float(residual_sigma),
            "worst_pair_p95_ms": float(worst_pair_p95),
            "convergence_time_s": float(since_s) if converged else 0.0,
            "freshness_max_age_s": float(max_age),
            # Row B – Mechanik
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
    # update age = max age across nodes based on mesh_clock
    with get_conn() as conn:
        t0 = now_s() - WINDOW_S
        max_age, _ = freshness_seconds(conn, t0)
    return jsonify({
        "app": "MeshTime v2",
        "window_s": WINDOW_S,
        "root_id": root,
        "update_age_s": max_age,
        "t_mesh_scale": "epoch_s",
        "build": "v2-dashboard",
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

    # histogram + per-node stats
    hist = {}
    stats = {}
    for nid, pts in series.items():
        xs = [v for (_t, v) in pts]
        if not xs:
            hist[nid] = {"bins": [], "counts": []}
            stats[nid] = {"mean": 0, "median": 0, "p95abs": 0, "sigma": 0}
            continue

        mean = float(sum(xs) / len(xs))
        med = median(xs)
        p95a = p95_abs_ms(xs)
        sig = robust_sigma_ms(xs)

        # histogram bins
        lo = min(xs); hi = max(xs)
        # expand a little for stable view
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
        stats[nid] = {"mean": mean, "median": med, "p95abs": p95a, "sigma": sig}

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

    # Plot 3A: innovation (use innov_med_ms; optionally also p95)
    by_node = {}
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

    by_node = {}
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

    # outlier fraction pNIS>9 (using median series)
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

    by_node = {}
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

    by_link = {}
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

    # compute latest sigma per link for topology coloring
    latest = {}
    for k, s in by_link.items():
        # choose last non-null sigma
        sigs = [x for x in s["sigma"] if x is not None]
        latest[k] = float(sigs[-1]) if sigs else None

    return jsonify({"series": by_link, "latest_sigma_ms": latest})

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
    body { margin: 0; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; background: #0b0f14; color: #e6edf3; }
    .wrap { padding: 14px 16px 18px; }
    .header { display:flex; gap:12px; align-items:center; justify-content:space-between; padding: 10px 12px; border:1px solid #1f2a37; border-radius: 12px; background:#0e141b; }
    .title { font-weight: 700; letter-spacing: .2px; }
    .meta { color:#9fb0c0; font-size: 12px; white-space: nowrap; }
    .ampel { width: 12px; height: 12px; border-radius: 50%; display:inline-block; margin-right: 10px; }
    .diag { display:flex; align-items:center; gap:8px; font-size: 13px; color:#cfe3f7; }
    .grid { margin-top: 12px; display:grid; grid-template-columns: 1.15fr 1fr; gap: 12px; }
    .grid2 { margin-top: 12px; display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .grid3 { margin-top: 12px; display:grid; grid-template-columns: 1fr 1fr 0.85fr; gap: 12px; }
    .card { border:1px solid #1f2a37; border-radius: 12px; background:#0e141b; padding: 10px 12px; }
    .card h3 { margin: 0 0 10px 0; font-size: 13px; color:#cfe3f7; letter-spacing: .2px; }
    .kpis { display:grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
    .kpi { border:1px solid #162130; border-radius: 10px; padding: 8px 10px; background:#0b1118; }
    .kpi .k { font-size: 11px; color:#9fb0c0; }
    .kpi .v { font-size: 16px; font-weight: 700; margin-top: 2px; }
    .kpi .s { font-size: 11px; color:#9fb0c0; margin-top: 2px; }
    canvas { width: 100% !important; height: 240px !important; }
    .small canvas { height: 190px !important; }
    .tiny canvas { height: 160px !important; }
    .row { display:flex; gap:12px; }
    .col { flex:1; }
    .mini { font-size: 11px; color:#9fb0c0; }
    #topo { width: 100%; height: 360px; border-radius: 10px; background:#0b1118; border:1px solid #162130; }
    .twoPlot { display:grid; grid-template-columns: 1.2fr 0.8fr; gap: 10px; align-items: start; }
    .statsBox { border:1px solid #162130; border-radius:10px; background:#0b1118; padding: 8px 10px; }
    .statsBox pre { margin:0; font-size: 11px; color:#cfe3f7; white-space: pre-wrap; }
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
        <div><canvas id="plotDelta"></canvas></div>
        <div class="statsBox"><pre id="deltaStats">…</pre></div>
      </div>
    </div>
    <div class="card small">
      <h3>Δ Distribution (10min)</h3>
      <canvas id="plotHist"></canvas>
      <div class="mini" id="histMini"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h3>Innovation (ms)</h3>
      <canvas id="plotInnov"></canvas>
    </div>
    <div class="card">
      <h3>pNIS (dimensionless) · Outlier rate</h3>
      <canvas id="plotPNIS"></canvas>
      <div class="mini" id="pnisMini"></div>
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h3>Controller Effort · desired vs applied (ms)</h3>
      <canvas id="plotCtrl"></canvas>
    </div>
    <div class="card">
      <h3>Saturation · clip / η</h3>
      <canvas id="plotClip"></canvas>
      <div class="mini" id="clipMini"></div>
    </div>
  </div>

  <div class="grid3">
    <div class="card">
      <h3>Link Quality · θ (ms)</h3>
      <canvas id="plotTheta"></canvas>
    </div>
    <div class="card">
      <h3>Link Quality · RTT (ms) / σ_link (ms)</h3>
      <canvas id="plotRTT"></canvas>
    </div>
    <div class="card tiny">
      <h3>Topology</h3>
      <canvas id="topo"></canvas>
      <div class="mini" id="topoMini"></div>
    </div>
  </div>
</div>

<script>
const fmt = (x, d=2) => (x===null||x===undefined||Number.isNaN(x)) ? "—" : Number(x).toFixed(d);
const fmtInt = (x) => (x===null||x===undefined||Number.isNaN(x)) ? "—" : String(Math.round(Number(x)));

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

function mkLine(ctx, datasets, yTitle){
  return new Chart(ctx, {
    type:"line",
    data:{ datasets },
    options:{
      responsive:true,
      animation:false,
      parsing:false,
      scales:{
        x:{ type:"linear", ticks:{ color:"#9fb0c0" }, grid:{ color:"#162130" } },
        y:{ ticks:{ color:"#9fb0c0" }, grid:{ color:"#162130" }, title:{ display:true, text:yTitle, color:"#9fb0c0" } }
      },
      plugins:{ legend:{ labels:{ color:"#cfe3f7" } } },
      elements:{ point:{ radius:0 } }
    }
  });
}

function mkBar(ctx, labels, datasets, yTitle){
  return new Chart(ctx, {
    type:"bar",
    data:{ labels, datasets },
    options:{
      responsive:true,
      animation:false,
      scales:{
        x:{ ticks:{ color:"#9fb0c0" }, grid:{ color:"#162130" } },
        y:{ ticks:{ color:"#9fb0c0" }, grid:{ color:"#162130" }, title:{ display:true, text:yTitle, color:"#9fb0c0" } }
      },
      plugins:{ legend:{ labels:{ color:"#cfe3f7" } } }
    }
  });
}

let C_delta=null, C_hist=null, C_innov=null, C_pnis=null, C_ctrl=null, C_clip=null, C_theta=null, C_rtt=null;

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

  // Mechanik row
  // innovation worst
  B.appendChild(kpiBox("Innovation |med| worst (ms)", fmt(k.innovation_med_abs_ms_worst,2)));
  // pNIS worst (median)
  const pnisVals = Object.values(k.pnis_med_by_node || {});
  B.appendChild(kpiBox("Consistency pNIS med", pnisVals.length? fmt(pnisVals.sort((a,b)=>b-a)[0],2):"—"));
  // saturation worst
  const clipVals = k.clip_rate_by_node || {};
  const clipWorst = Object.keys(clipVals).length ? Math.max(...Object.values(clipVals)) : null;
  B.appendChild(kpiBox("Controller saturation (clip %)", clipWorst!==null? fmt(clipWorst,0):"—"));
  // link noise med + worst
  B.appendChild(kpiBox("Link σ_med (ms)", fmt(k.link_sigma_med_overall_ms,2), `worst ${fmt(k.link_sigma_worst_ms,2)}`));

  // Panel 2A: delta to root
  const dsDelta = [];
  const statsLines = [];
  for(const [nid, s] of Object.entries(rel.series)){
    const pts = s.t.map((t,i)=>({x:t, y:s.y[i]}));
    dsDelta.push({ label:nid, data:pts });
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

  // Panel 2B: histogram (stacked bars per node)
  // Use first node's bins as x-labels (they differ slightly; good enough for glance)
  const nodes = Object.keys(rel.hist || {});
  let labels = [];
  if(nodes.length){
    labels = (rel.hist[nodes[0]].bins || []).map(x=>fmt(x,1));
  }
  const dsH = [];
  for(const nid of nodes){
    const h = rel.hist[nid];
    dsH.push({ label:nid, data: h.counts || [] });
  }
  if(!C_hist){
    C_hist = mkBar(document.getElementById("plotHist"), labels, dsH, "count");
  }else{
    C_hist.data.labels = labels;
    C_hist.data.datasets = dsH;
    C_hist.update();
  }
  document.getElementById("histMini").textContent = nodes.length ? `Histogram bins ≈ ${nodes.length} nodes` : "no data";

  // Panel 3A: innovation
  const dsInnov = [];
  for(const [nid, s] of Object.entries(innov)){
    const pts = s.t.map((t,i)=>({x:t, y:s.innov[i]}));
    dsInnov.push({ label:nid, data:pts });
  }
  if(!C_innov){
    C_innov = mkLine(document.getElementById("plotInnov"), dsInnov, "innov (ms)");
  }else{
    C_innov.data.datasets = dsInnov;
    C_innov.update();
  }

  // Panel 3B: pNIS
  const dsPN = [];
  for(const [nid, s] of Object.entries(pnis.series || {})){
    const pts = s.t.map((t,i)=>({x:t, y:s.pnis[i]}));
    dsPN.push({ label:nid, data:pts });
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

  // Panel 4A: controller desired vs applied (overlay as separate datasets per node)
  const dsCtrl = [];
  for(const [nid, s] of Object.entries(ctrl)){
    const ptsD = s.t.map((t,i)=>({x:t, y:s.desired[i]}));
    const ptsA = s.t.map((t,i)=>({x:t, y:s.applied[i]}));
    dsCtrl.push({ label:`${nid} desired`, data:ptsD });
    dsCtrl.push({ label:`${nid} applied`, data:ptsA });
  }
  if(!C_ctrl){
    C_ctrl = mkLine(document.getElementById("plotCtrl"), dsCtrl, "delta (ms)");
  }else{
    C_ctrl.data.datasets = dsCtrl;
    C_ctrl.update();
  }

  // Panel 4B: clip (0/1) and eta (we plot clip as line, eta as line)
  const dsClip = [];
  const clipLines = [];
  for(const [nid, s] of Object.entries(ctrl)){
    const ptsC = s.t.map((t,i)=>({x:t, y:s.clip[i]}));
    dsClip.push({ label:`${nid} clip`, data:ptsC });
    const etaPts = s.t.map((t,i)=>({x:t, y:(s.eta[i]===null? null : s.eta[i])})).filter(p=>p.y!==null);
    dsClip.push({ label:`${nid} η`, data:etaPts });
    // quick clip rate + eta med
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
  for(const [lk, s] of Object.entries(ls)){
    const th = s.t.map((t,i)=>({x:t, y:s.theta[i]})).filter(p=>p.y!==null);
    const rttPts = s.t.map((t,i)=>({x:t, y:s.rtt[i]})).filter(p=>p.y!==null);
    const sigPts = s.t.map((t,i)=>({x:t, y:s.sigma[i]})).filter(p=>p.y!==null);
    dsTheta.push({ label:lk, data:th });
    dsRTT.push({ label:`${lk} RTT`, data:rttPts });
    dsRTT.push({ label:`${lk} σ`, data:sigPts });
  }
  if(!C_theta){
    C_theta = mkLine(document.getElementById("plotTheta"), dsTheta, "θ (ms)");
  }else{
    C_theta.data.datasets = dsTheta;
    C_theta.update();
  }
  if(!C_rtt){
    C_rtt = mkLine(document.getElementById("plotRTT"), dsRTT, "RTT / σ (ms)");
  }else{
    C_rtt.data.datasets = dsRTT;
    C_rtt.update();
  }

  // Topology (tiny canvas, simple radial layout)
  drawTopo(topo, link.latest_sigma_ms || {});
}

function drawTopo(topo, latestSig){
  const canvas = document.getElementById("topo");
  const ctx = canvas.getContext("2d");
  const W = canvas.width = canvas.clientWidth * devicePixelRatio;
  const H = canvas.height = canvas.clientHeight * devicePixelRatio;
  ctx.clearRect(0,0,W,H);

  const nodes = topo.nodes || [];
  const links = topo.links || [];
  const root = topo.root_id;

  const cx = W/2, cy = H/2;
  const R = Math.min(W,H)*0.33;

  // positions
  const ids = nodes.map(n=>n.id);
  const n = ids.length || 1;
  const pos = {};
  nodes.forEach((nd,i)=>{
    const ang = (2*Math.PI*i)/n - Math.PI/2;
    const rr = (nd.id===root) ? 0 : R;
    pos[nd.id] = {x: cx + rr*Math.cos(ang), y: cy + rr*Math.sin(ang), color: nd.color || "#3498db", is_root: nd.id===root};
  });

  // links: thickness by sigma
  const sigVals = Object.values(latestSig).filter(x=>x!==null && x!==undefined);
  const sigMed = sigVals.length ? sigVals.sort((a,b)=>a-b)[Math.floor(sigVals.length/2)] : 1.0;

  links.forEach(l=>{
    const a = pos[l.source], b = pos[l.target];
    if(!a || !b) return;
    const k = `${l.source}→${l.target}`;
    const s = latestSig[k];
    let w = 1.0;
    if(s!==null && s!==undefined){
      w = 1.0 + 3.0 * Math.min(3.0, (s / Math.max(0.5, sigMed)));
    }
    ctx.strokeStyle = "rgba(159,176,192,0.45)";
    ctx.lineWidth = w * devicePixelRatio;
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
  });

  // nodes
  nodes.forEach(nd=>{
    const p = pos[nd.id];
    if(!p) return;
    ctx.fillStyle = p.is_root ? "rgba(46,204,113,0.95)" : "rgba(52,152,219,0.95)";
    ctx.beginPath(); ctx.arc(p.x,p.y, (p.is_root?8:7)*devicePixelRatio, 0, 2*Math.PI); ctx.fill();
    ctx.fillStyle = "rgba(230,237,243,0.9)";
    ctx.font = `${12*devicePixelRatio}px system-ui`;
    ctx.fillText(nd.id, p.x+10*devicePixelRatio, p.y+4*devicePixelRatio);
  });

  document.getElementById("topoMini").textContent = `root=${root} · links=${links.length}`;
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
    # Ensure DB exists
    if not DB_PATH.exists():
        raise SystemExit(f"DB not found: {DB_PATH}")
    app.run(host="0.0.0.0", port=8000, debug=False)

if __name__ == "__main__":
    main()
