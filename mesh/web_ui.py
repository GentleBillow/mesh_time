# -*- coding: utf-8 -*-
# mesh/web_ui.py
#
# MeshTime Dashboard (Single Page) — Operator + Debug
#
# Principles:
#   - x-axis is created_at (sink clock)
#   - "theoretisch optimale Meshzeit" = robust consensus = median(t_mesh across nodes) per time bin
#   - primary: centered mesh error ε_i(t) = t_mesh(i,t) - consensus(t)   (ms)
#   - never crash on missing columns / None
#   - NO "ms twice" bug: robust unit inference + consistent scaling

from __future__ import annotations

import json
import math
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flask import Flask, jsonify, make_response, render_template_string

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "mesh_data.sqlite"
CFG_PATH = BASE_DIR / "config" / "nodes.json"

from mesh.storage import Storage  # ensures schema exists

app = Flask(__name__)

# -----------------------------
# UI tuning
# -----------------------------
WINDOW_SECONDS_DEFAULT = 600.0
BIN_S_DEFAULT = 0.5
MAX_NODE_POINTS_DEFAULT = 20000
MAX_LINK_POINTS_DEFAULT = 30000
HEATMAP_MAX_BINS = 60

# Convergence thresholds
CONV_WINDOW_S = 30.0
MIN_SAMPLES_WARMUP = 8

THRESH_DELTA_APPLIED_MED_MS = 0.5
THRESH_SLEW_CLIP_RATE = 0.20
THRESH_LINK_SIGMA_MED_MS = 2.0

FRESH_MIN_S = 3.0
FRESH_MULT = 6.0

K_STABLE_LINKS_DEFAULT = 1


# -----------------------------
# Helpers (must never crash)
# -----------------------------
def _json_error(msg: str, status: int = 500):
    return make_response(jsonify({"error": str(msg)}), int(status))


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _table_cols(conn: sqlite3.Connection, table: str) -> Set[str]:
    try:
        cur = conn.cursor()
        return {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def row_has(r: Any, key: str) -> bool:
    try:
        return key in r.keys()  # sqlite3.Row
    except Exception:
        try:
            return key in r
        except Exception:
            return False


def row_get(r: Any, key: str, default: Any = None) -> Any:
    if not row_has(r, key):
        return default
    try:
        return r[key]
    except Exception:
        return default


def _f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _utc(ts: Optional[float]) -> str:
    if ts is None:
        return "n/a"
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


def _quantile_sorted(xs: List[float], q: float) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(xs[0])
    pos = float(q) * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def robust_median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    return float(_quantile_sorted(xs, 0.5))


def robust_iqr(values: List[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    return float(_quantile_sorted(xs, 0.75) - _quantile_sorted(xs, 0.25))


def load_config() -> Dict[str, Any]:
    try:
        if not CFG_PATH.exists():
            return {}
        with CFG_PATH.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def get_topology() -> Dict[str, Any]:
    cfg = load_config()
    nodes = []
    links = []

    for node_id, entry in (cfg.items() if isinstance(cfg, dict) else []):
        if node_id == "sync":
            continue
        if not isinstance(entry, dict):
            continue

        ip = entry.get("ip")
        color = entry.get("color") or "#3498db"
        sync_cfg = entry.get("sync", {}) or {}
        is_root = bool(sync_cfg.get("is_root", False))

        nodes.append({"id": node_id, "ip": ip, "color": color, "is_root": is_root})

        neighs = entry.get("neighbors", []) or []
        for neigh in neighs:
            try:
                neigh = str(neigh)
            except Exception:
                continue
            if node_id < neigh:
                links.append({"source": node_id, "target": neigh})

    return {"nodes": nodes, "links": links}


def _beacon_period_s(cfg: Dict[str, Any]) -> float:
    try:
        sync = cfg.get("sync", {}) or {}
        return float(sync.get("beacon_period_s", 0.5))
    except Exception:
        return 0.5


def _fresh_thresh_s(cfg: Dict[str, Any]) -> float:
    bp = _beacon_period_s(cfg)
    return max(FRESH_MIN_S, bp * FRESH_MULT)


def _k_stable_links(cfg: Dict[str, Any]) -> int:
    try:
        sync = cfg.get("sync", {}) or {}
        k = int(sync.get("ui_k_stable_links", K_STABLE_LINKS_DEFAULT))
        return max(0, k)
    except Exception:
        return K_STABLE_LINKS_DEFAULT


# -----------------------------
# DB readers
# -----------------------------
def fetch_node_rows(window_s: float, limit: int) -> List[sqlite3.Row]:
    conn = get_conn()
    try:
        cols = _table_cols(conn, "ntp_reference")
        if not {"node_id", "created_at"}.issubset(cols):
            return []

        wanted = [
            "id", "node_id", "created_at",
            "t_mesh", "offset", "err_mesh_vs_wall",
            "delta_desired_ms", "delta_applied_ms", "dt_s", "slew_clipped",
        ]
        sel = [c for c in wanted if c in cols]
        if not sel:
            return []

        cutoff = time.time() - float(window_s)
        peer_filter = "AND peer_id IS NULL" if "peer_id" in cols else ""
        q = f"""
            SELECT {", ".join(sel)}
            FROM ntp_reference
            WHERE created_at >= ?
            {peer_filter}
            ORDER BY created_at ASC
            LIMIT ?
        """
        cur = conn.cursor()
        return cur.execute(q, (cutoff, int(limit))).fetchall()
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def fetch_link_rows(window_s: float, limit: int) -> List[sqlite3.Row]:
    conn = get_conn()
    try:
        cols = _table_cols(conn, "ntp_reference")
        need = {"node_id", "peer_id", "created_at"}
        if not need.issubset(cols):
            return []

        wanted = ["id", "node_id", "peer_id", "created_at", "theta_ms", "rtt_ms", "sigma_ms"]
        sel = [c for c in wanted if c in cols]
        if "peer_id" not in sel:
            return []

        cutoff = time.time() - float(window_s)
        q = f"""
            SELECT {", ".join(sel)}
            FROM ntp_reference
            WHERE created_at >= ?
              AND peer_id IS NOT NULL
            ORDER BY created_at ASC
            LIMIT ?
        """
        cur = conn.cursor()
        return cur.execute(q, (cutoff, int(limit))).fetchall()
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


# -----------------------------
# Unit inference (fixes "ms twice")
# -----------------------------
def infer_time_unit_per_second_from_deltas(rows: List[sqlite3.Row], value_col: str) -> Optional[float]:
    """
    Infer unit factor relative to seconds based on median ratio:
      ratio = Δvalue / Δcreated_at

    If value is:
      - seconds: ratio ~ 1
      - milliseconds: ratio ~ 1000
      - microseconds: ratio ~ 1e6

    Returns the snapped unit_per_second in {1, 1e3, 1e6} if confident, else None.
    """
    by_node: Dict[str, List[Tuple[float, float]]] = {}

    for r in rows:
        nid = str(row_get(r, "node_id", "") or "")
        tc = _f(row_get(r, "created_at"))
        vv = _f(row_get(r, value_col))
        if not nid or tc is None or vv is None:
            continue
        by_node.setdefault(nid, []).append((tc, vv))

    ratios: List[float] = []
    for pts in by_node.values():
        pts.sort(key=lambda x: x[0])
        for i in range(1, len(pts)):
            t0, v0 = pts[i - 1]
            t1, v1 = pts[i]
            dt = float(t1 - t0)
            dv = float(v1 - v0)
            if dt <= 1e-6:
                continue
            # ignore tiny dv and resets
            if abs(dv) <= 0:
                continue
            ratio = dv / dt
            if ratio <= 0:
                continue
            # plausible ranges to avoid glitches nuking the median
            if ratio < 1e-3 or ratio > 1e9:
                continue
            ratios.append(ratio)

    r_med = robust_median(ratios)
    if r_med is None or r_med <= 0:
        return None

    # snap in log-space
    candidates = [1.0, 1e3, 1e6]
    best = min(candidates, key=lambda c: abs(math.log10(r_med) - math.log10(c)))

    # accept only if within ~0.35 decades (~2.2x)
    if abs(math.log10(r_med) - math.log10(best)) <= 0.35:
        return float(best)
    return None


def scale_value_to_ms(rows: List[sqlite3.Row], value_col: str) -> float:
    """
    Returns scale such that: value_ms = value_raw * scale

    For t_mesh: inferred via delta-ratio (best).
    For offset/err_mesh_vs_wall: try delta-ratio if possible, else magnitude heuristic.
    """
    unit_per_s = infer_time_unit_per_second_from_deltas(rows, value_col)

    if unit_per_s is not None:
        # if value unit is (unit_per_s * seconds), then ms = value * (1000 / unit_per_s)
        return float(1000.0 / unit_per_s)

    # fallback heuristic by magnitude of absolute median
    vals = []
    for r in rows:
        v = _f(row_get(r, value_col))
        if v is None:
            continue
        vals.append(abs(float(v)))
    med = robust_median(vals) if vals else None
    if med is None:
        # safest default for time-like fields in this project is seconds
        return 1000.0

    # typical offsets/errors are small if in seconds (<10), huge if in us
    if med >= 1e5:
        return 0.001  # us -> ms
    if med >= 1e2:
        return 1.0    # ms -> ms
    return 1000.0     # s -> ms


# -----------------------------
# Aggregation: centered mesh + old node plots + pairs etc.
# -----------------------------
def build_mesh_diagnostics(window_s: float, bin_s: float, max_points: int) -> Dict[str, Any]:
    rows = fetch_node_rows(window_s=window_s, limit=max_points)
    if not rows:
        return {
            "mesh_series": {},
            "step_series": {},
            "pairs": {},
            "stability": {},
            "heatmap": {"data": [], "n_bins": 0},
            "offset_series": {},
            "err_mesh_vs_wall_series": {},
            "meta": {"window_s": window_s, "bin_s": bin_s, "x_axis": "created_at", "note": "no node-only data"},
        }

    if bin_s <= 0:
        bin_s = BIN_S_DEFAULT

    # scale inference (no ms-twice)
    scale_tmesh = scale_value_to_ms(rows, "t_mesh")
    scale_offset = scale_value_to_ms(rows, "offset") if any(row_get(r, "offset") is not None for r in rows) else 1000.0
    scale_err = scale_value_to_ms(rows, "err_mesh_vs_wall") if any(row_get(r, "err_mesh_vs_wall") is not None for r in rows) else 1000.0

    ts = []
    for r in rows:
        t = _f(row_get(r, "created_at"))
        if t is not None:
            ts.append(t)

    if not ts:
        return {
            "mesh_series": {},
            "step_series": {},
            "pairs": {},
            "stability": {},
            "heatmap": {"data": [], "n_bins": 0},
            "offset_series": {},
            "err_mesh_vs_wall_series": {},
            "meta": {"window_s": window_s, "bin_s": bin_s, "x_axis": "created_at", "note": "no timestamps"},
        }

    t_min = min(ts)

    # also export legacy series (offset/err_mesh_vs_wall), raw timeline (no binning needed)
    offset_series: Dict[str, List[Dict[str, float]]] = {}
    err_series: Dict[str, List[Dict[str, float]]] = {}

    for r in rows:
        nid = str(row_get(r, "node_id", "") or "")
        t = _f(row_get(r, "created_at"))
        if not nid or t is None:
            continue

        off = _f(row_get(r, "offset"))
        if off is not None:
            offset_series.setdefault(nid, []).append({"t_wall": float(t), "offset_ms": float(off * scale_offset)})

        emvw = _f(row_get(r, "err_mesh_vs_wall"))
        if emvw is not None:
            err_series.setdefault(nid, []).append({"t_wall": float(t), "err_mesh_vs_wall_ms": float(emvw * scale_err)})

    for nid in offset_series:
        offset_series[nid].sort(key=lambda p: p["t_wall"])
    for nid in err_series:
        err_series[nid].sort(key=lambda p: p["t_wall"])

    # bin node t_mesh for consensus
    # bins[idx][node] = (t_last, tmesh_ms)
    bins: Dict[int, Dict[str, Tuple[float, float]]] = {}
    for r in rows:
        nid = str(row_get(r, "node_id", "") or "")
        t = _f(row_get(r, "created_at"))
        tm = _f(row_get(r, "t_mesh"))
        if not nid or t is None or tm is None:
            continue
        idx = int((t - t_min) / bin_s)
        if idx < 0:
            continue
        bucket = bins.setdefault(idx, {})
        prev = bucket.get(nid)
        if (prev is None) or (t >= prev[0]):
            bucket[nid] = (t, float(tm) * scale_tmesh)

    # compute centered ε per bin
    mesh_series: Dict[str, List[Dict[str, float]]] = {}
    for idx in sorted(bins.keys()):
        bucket = bins[idx]
        if len(bucket) < 2:
            continue
        consensus_ms = robust_median([v for (_t, v) in bucket.values()])
        if consensus_ms is None:
            continue
        t_bin = t_min + (idx + 0.5) * bin_s
        for nid, (_t, tmesh_ms) in bucket.items():
            mesh_series.setdefault(nid, []).append({"t_wall": float(t_bin), "mesh_err_ms": float(tmesh_ms - consensus_ms)})

    # per-node step series (Δε)
    step_series: Dict[str, List[Dict[str, float]]] = {}
    for nid, pts in mesh_series.items():
        pts.sort(key=lambda p: p["t_wall"])
        prev = None
        out = []
        for p in pts:
            cur = float(p["mesh_err_ms"])
            step = 0.0 if prev is None else float(cur - prev)
            out.append({"t_wall": float(p["t_wall"]), "step_ms": float(step)})
            prev = cur
        step_series[nid] = out

    # pairwise series using ε values per bin
    def norm_pair(a: str, b: str) -> str:
        return "-".join(sorted([a, b]))

    pairs: Dict[str, List[Dict[str, float]]] = {}
    for idx in sorted(bins.keys()):
        bucket = bins[idx]
        if len(bucket) < 2:
            continue
        consensus_ms = robust_median([v for (_t, v) in bucket.values()])
        if consensus_ms is None:
            continue
        eps = {nid: (tmesh_ms - consensus_ms) for nid, (_t, tmesh_ms) in bucket.items()}
        nodes_now = sorted(eps.keys())
        t_bin = t_min + (idx + 0.5) * bin_s
        for i in range(len(nodes_now)):
            for j in range(i + 1, len(nodes_now)):
                a = nodes_now[i]
                b = nodes_now[j]
                pid = norm_pair(a, b)
                delta = float(eps[a] - eps[b])
                pairs.setdefault(pid, []).append({"t_wall": float(t_bin), "delta_ms": delta, "bin": float(idx)})

    # stability per pair: robust sigma ≈ 0.7413*IQR
    stability: Dict[str, Dict[str, float]] = {}
    for pid, pts in pairs.items():
        deltas = [float(p["delta_ms"]) for p in pts if p.get("delta_ms") is not None]
        if len(deltas) < 10:
            continue
        iqr = robust_iqr(deltas)
        sigma = 0.7413 * iqr
        stability[pid] = {"sigma_ms": float(sigma), "iqr_ms": float(iqr), "n": float(len(deltas))}

    # heatmap: binned |pair delta|
    heatmap_data: List[Dict[str, Any]] = []
    n_bins = 0
    if pairs and bins:
        idx_min = min(bins.keys())
        idx_max = max(bins.keys())
        total_bins = max(1, idx_max - idx_min + 1)
        display_bins = min(HEATMAP_MAX_BINS, total_bins)

        if display_bins == total_bins:
            for pid, pts in pairs.items():
                for p in pts:
                    heatmap_data.append({"pair": pid, "t_bin": float(p["t_wall"]), "value": abs(float(p["delta_ms"]))})
            n_bins = total_bins
        else:
            accum: Dict[Tuple[str, int], List[float]] = {}
            for pid, pts in pairs.items():
                for p in pts:
                    bi = int(p.get("bin", idx_min))
                    bi = max(idx_min, min(idx_max, bi))
                    rel = (bi - idx_min) / max(1.0, float(total_bins))
                    d_idx = int(rel * display_bins)
                    if d_idx >= display_bins:
                        d_idx = display_bins - 1
                    accum.setdefault((pid, d_idx), []).append(abs(float(p["delta_ms"])))

            for (pid, d_idx), vals in accum.items():
                idx_center = idx_min + (d_idx + 0.5) * (total_bins / display_bins)
                t_center = t_min + (idx_center + 0.5) * bin_s
                heatmap_data.append({"pair": pid, "t_bin": float(t_center), "value": float(sum(vals) / max(1, len(vals)))})
            n_bins = display_bins

    return {
        "mesh_series": mesh_series,
        "step_series": step_series,
        "pairs": pairs,
        "stability": stability,
        "heatmap": {"data": heatmap_data, "n_bins": int(n_bins)},
        "offset_series": offset_series,
        "err_mesh_vs_wall_series": err_series,
        "meta": {
            "window_s": float(window_s),
            "bin_s": float(bin_s),
            "x_axis": "created_at",
            "scale": {
                "t_mesh_to_ms": float(scale_tmesh),
                "offset_to_ms": float(scale_offset),
                "err_mesh_vs_wall_to_ms": float(scale_err),
            },
        },
    }


# -----------------------------
# Link diagnostics
# -----------------------------
def build_link_diagnostics(window_s: float, bin_s: float, max_points: int) -> Dict[str, Any]:
    rows = fetch_link_rows(window_s=window_s, limit=max_points)
    if not rows:
        return {"links": {}, "latest_sigma": {}, "meta": {"window_s": window_s, "bin_s": bin_s, "x_axis": "created_at"}}

    if bin_s <= 0:
        bin_s = BIN_S_DEFAULT

    ts = []
    for r in rows:
        t = _f(row_get(r, "created_at"))
        if t is not None:
            ts.append(t)

    if not ts:
        return {"links": {}, "latest_sigma": {}, "meta": {"window_s": window_s, "bin_s": bin_s, "x_axis": "created_at"}}

    t_min = min(ts)

    def lid(a: str, b: str) -> str:
        return f"{a}->{b}"

    # bins[idx][link] = (t, theta, rtt, sigma)
    bins: Dict[int, Dict[str, Tuple[float, Optional[float], Optional[float], Optional[float]]]] = {}
    for r in rows:
        t = _f(row_get(r, "created_at"))
        a = str(row_get(r, "node_id", "") or "")
        b = str(row_get(r, "peer_id", "") or "")
        if t is None or not a or not b:
            continue
        theta = _f(row_get(r, "theta_ms"))
        rtt = _f(row_get(r, "rtt_ms"))
        sigma = _f(row_get(r, "sigma_ms"))

        idx = int((t - t_min) / bin_s)
        if idx < 0:
            continue
        bucket = bins.setdefault(idx, {})
        prev = bucket.get(lid(a, b))
        if (prev is None) or (t >= prev[0]):
            bucket[lid(a, b)] = (t, theta, rtt, sigma)

    links: Dict[str, List[Dict[str, Any]]] = {}
    latest_sigma: Dict[str, float] = {}
    for idx in sorted(bins.keys()):
        t_bin = t_min + (idx + 0.5) * bin_s
        for link_id, (_t, theta, rtt, sigma) in bins[idx].items():
            obj: Dict[str, Any] = {"t_wall": float(t_bin)}
            if theta is not None:
                obj["theta_ms"] = float(theta)
            if rtt is not None:
                obj["rtt_ms"] = float(rtt)
            if sigma is not None:
                obj["sigma_ms"] = float(sigma)
                latest_sigma[link_id] = float(sigma)
            links.setdefault(link_id, []).append(obj)

    for link_id, pts in links.items():
        pts.sort(key=lambda p: p["t_wall"])

    return {"links": links, "latest_sigma": latest_sigma, "meta": {"window_s": float(window_s), "bin_s": float(bin_s), "x_axis": "created_at"}}


# -----------------------------
# Controller timeseries
# -----------------------------
def build_controller_timeseries(window_s: float, max_points: int) -> Dict[str, Any]:
    rows = fetch_node_rows(window_s=window_s, limit=max_points)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        nid = str(row_get(r, "node_id", "") or "")
        t = _f(row_get(r, "created_at"))
        if not nid or t is None:
            continue
        obj: Dict[str, Any] = {"t_wall": float(t)}

        for k in ["delta_desired_ms", "delta_applied_ms", "dt_s"]:
            v = _f(row_get(r, k))
            if v is not None:
                obj[k] = float(v)

        sc = row_get(r, "slew_clipped")
        if sc is not None:
            try:
                obj["slew_clipped"] = 1 if int(sc) else 0
            except Exception:
                pass

        out.setdefault(nid, []).append(obj)

    for nid, pts in out.items():
        pts.sort(key=lambda p: p["t_wall"])
    return {"controller": out, "meta": {"window_s": float(window_s), "x_axis": "created_at"}}


# -----------------------------
# Overview / Status (optimal mesh time, not vs C) + stable links rule
# -----------------------------
def compute_overview(window_s: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    eff = max(float(window_s), float(CONV_WINDOW_S))

    node_rows = fetch_node_rows(window_s=eff, limit=MAX_NODE_POINTS_DEFAULT)
    link_rows = fetch_link_rows(window_s=eff, limit=MAX_LINK_POINTS_DEFAULT)

    fresh_thr = _fresh_thresh_s(cfg)
    k_stable = _k_stable_links(cfg)

    latest: Dict[str, sqlite3.Row] = {}
    by_node: Dict[str, List[sqlite3.Row]] = {}
    for r in node_rows:
        nid = str(row_get(r, "node_id", "") or "")
        if not nid:
            continue
        by_node.setdefault(nid, []).append(r)
        latest[nid] = r

    # consensus now based on latest t_mesh (scaled correctly)
    scale_tmesh = scale_value_to_ms(list(latest.values()), "t_mesh") if latest else 1000.0
    tmesh_now_ms: List[float] = []
    tmesh_now_map: Dict[str, float] = {}
    for nid, r in latest.items():
        tm = _f(row_get(r, "t_mesh"))
        if tm is not None:
            v_ms = float(tm) * scale_tmesh
            tmesh_now_ms.append(v_ms)
            tmesh_now_map[nid] = v_ms

    consensus_now_ms = robust_median(tmesh_now_ms) if tmesh_now_ms else None

    # link sigma per directed link within conv window
    conv_cut = now - float(CONV_WINDOW_S)
    sigs_by_link: Dict[str, List[float]] = {}
    last_seen_link: Dict[str, float] = {}
    for r in link_rows:
        t = _f(row_get(r, "created_at"))
        a = str(row_get(r, "node_id", "") or "")
        b = str(row_get(r, "peer_id", "") or "")
        if t is None or not a or not b:
            continue
        lid = f"{a}->{b}"
        last_seen_link[lid] = max(last_seen_link.get(lid, 0.0), float(t))
        if t >= conv_cut:
            sg = _f(row_get(r, "sigma_ms"))
            if sg is not None:
                sigs_by_link.setdefault(lid, []).append(float(sg))

    sigma_med_link: Dict[str, Optional[float]] = {lid: robust_median(vals) for lid, vals in sigs_by_link.items()}

    nodes_out: List[Dict[str, Any]] = []
    offenders = {"worst_node_correction": None, "stalest_node": None, "worst_link_sigma": None, "most_slew_clipped": None}

    worst_corr = -1.0
    worst_age = -1.0
    worst_sigma = -1.0
    worst_clip = -1.0

    for nid in sorted(latest.keys()):
        r_last = latest[nid]
        t_last = _f(row_get(r_last, "created_at"))
        age_s = (now - t_last) if t_last is not None else None

        mesh_err_now_ms = None
        if consensus_now_ms is not None and nid in tmesh_now_map:
            mesh_err_now_ms = float(tmesh_now_map[nid] - consensus_now_ms)

        recent = []
        for rr in by_node.get(nid, []):
            t = _f(row_get(rr, "created_at"))
            if t is not None and t >= conv_cut:
                recent.append(rr)

        abs_applied = []
        clipped = []
        for rr in recent:
            da = _f(row_get(rr, "delta_applied_ms"))
            if da is not None:
                abs_applied.append(abs(float(da)))
            sc = row_get(rr, "slew_clipped")
            try:
                if sc is not None:
                    clipped.append(1 if int(sc) else 0)
            except Exception:
                pass

        med_abs_applied = robust_median(abs_applied)
        clip_rate = (sum(clipped) / len(clipped)) if clipped else None

        mesh_rate_ms_s = None
        if len(recent) >= 2:
            r0 = recent[-2]
            r1 = recent[-1]
            t0 = _f(row_get(r0, "created_at"))
            t1 = _f(row_get(r1, "created_at"))
            tm0 = _f(row_get(r0, "t_mesh"))
            tm1 = _f(row_get(r1, "t_mesh"))
            if t0 is not None and t1 is not None and tm0 is not None and tm1 is not None and t1 > t0:
                mesh_rate_ms_s = float(((tm1 - tm0) * scale_tmesh) / (t1 - t0))

        # stable links rule (B): require at least K stable outgoing links
        stable_links = 0
        total_links_considered = 0
        for lid, med_sig in sigma_med_link.items():
            if not lid.startswith(nid + "->"):
                continue
            ls = last_seen_link.get(lid)
            link_age = (now - ls) if ls is not None else None
            if link_age is None or link_age > fresh_thr:
                continue
            total_links_considered += 1
            if med_sig is not None and med_sig <= THRESH_LINK_SIGMA_MED_MS:
                stable_links += 1

        enough = len(recent) >= MIN_SAMPLES_WARMUP
        fresh_ok = (age_s is not None and age_s <= fresh_thr)
        corr_ok = (med_abs_applied is not None and med_abs_applied <= THRESH_DELTA_APPLIED_MED_MS)
        clip_ok = (clip_rate is None) or (clip_rate <= THRESH_SLEW_CLIP_RATE)
        link_ok = (k_stable <= 0) or (stable_links >= k_stable)

        if not enough:
            state = "YELLOW"
            reason = "warming up"
        elif not fresh_ok:
            state = "RED"
            reason = f"stale data (age {age_s:.1f}s)" if age_s is not None else "stale data"
        elif corr_ok and clip_ok and link_ok:
            state = "GREEN"
            reason = "converged"
        else:
            state = "YELLOW"
            rs = []
            if not corr_ok:
                rs.append(f"|Δapplied|med {med_abs_applied:.2f}ms" if med_abs_applied is not None else "|Δapplied|med n/a")
            if not clip_ok and clip_rate is not None:
                rs.append(f"slew_clipped {clip_rate*100:.0f}%")
            if not link_ok:
                rs.append(f"stable_links {stable_links}/{k_stable}")
            reason = ", ".join(rs) if rs else "not converged"

        if med_abs_applied is not None and med_abs_applied > worst_corr:
            worst_corr = med_abs_applied
            offenders["worst_node_correction"] = nid
        if age_s is not None and age_s > worst_age:
            worst_age = age_s
            offenders["stalest_node"] = nid
        if clip_rate is not None and clip_rate > worst_clip:
            worst_clip = clip_rate
            offenders["most_slew_clipped"] = nid

        nodes_out.append({
            "node_id": nid,
            "state": state,
            "reason": reason,
            "last_seen_utc": _utc(t_last),
            "age_s": age_s,
            "mesh_err_now_ms": mesh_err_now_ms,
            "mesh_rate_ms_s": mesh_rate_ms_s,
            "med_abs_delta_applied_ms": med_abs_applied,
            "slew_clip_rate": clip_rate,
            "stable_links": stable_links,
            "k_stable_links": k_stable,
            "links_considered": total_links_considered,
        })

    for lid, med_sig in sigma_med_link.items():
        if med_sig is not None and med_sig > worst_sigma:
            worst_sigma = med_sig
            offenders["worst_link_sigma"] = lid

    mesh_state = "GREEN" if nodes_out and all(n["state"] == "GREEN" for n in nodes_out) else "YELLOW"
    if any(n["state"] == "RED" for n in nodes_out):
        mesh_state = "RED"

    if mesh_state == "GREEN":
        mesh_reason = "converged"
    else:
        parts = []
        if offenders["stalest_node"]:
            parts.append(f"stale {offenders['stalest_node']}")
        if offenders["worst_node_correction"]:
            parts.append(f"correction {offenders['worst_node_correction']}")
        if offenders["worst_link_sigma"]:
            parts.append(f"linkσ {offenders['worst_link_sigma']}")
        mesh_reason = ", ".join(parts) if parts else "not converged"

    return {
        "mesh": {
            "state": mesh_state,
            "reason": mesh_reason,
            "now_utc": _utc(now),
            "conv_window_s": float(CONV_WINDOW_S),
            "consensus_now_ms": consensus_now_ms,
            "scale_t_mesh_to_ms": float(scale_tmesh),
            "thresholds": {
                "fresh_s": float(fresh_thr),
                "delta_applied_med_ms": float(THRESH_DELTA_APPLIED_MED_MS),
                "slew_clip_rate": float(THRESH_SLEW_CLIP_RATE),
                "link_sigma_med_ms": float(THRESH_LINK_SIGMA_MED_MS),
                "k_stable_links": int(k_stable),
                "warmup_min_samples": int(MIN_SAMPLES_WARMUP),
            },
        },
        "nodes": nodes_out,
        "offenders": offenders,
        "link_sigma_med": {k: v for k, v in sigma_med_link.items() if v is not None},
        "link_last_seen": last_seen_link,
    }


# -----------------------------
# Routes
# -----------------------------
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


@app.route("/api/overview")
def api_overview():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync.get("ui_window_s", WINDOW_SECONDS_DEFAULT))
        return jsonify(compute_overview(window_s, cfg))
    except Exception as e:
        return _json_error(f"/api/overview failed: {e}", 500)


@app.route("/api/mesh_diag")
def api_mesh_diag():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync.get("ui_window_s", WINDOW_SECONDS_DEFAULT))
        bin_s = float(sync.get("ui_bin_s", BIN_S_DEFAULT))
        max_points = int(sync.get("ui_max_points", MAX_NODE_POINTS_DEFAULT))
        return jsonify(build_mesh_diagnostics(window_s=window_s, bin_s=bin_s, max_points=max_points))
    except Exception as e:
        return _json_error(f"/api/mesh_diag failed: {e}", 500)


@app.route("/api/link_diag")
def api_link_diag():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync.get("ui_window_s", WINDOW_SECONDS_DEFAULT))
        bin_s = float(sync.get("ui_bin_s", BIN_S_DEFAULT))
        max_points = int(sync.get("ui_link_max_points", MAX_LINK_POINTS_DEFAULT))
        return jsonify(build_link_diagnostics(window_s=window_s, bin_s=bin_s, max_points=max_points))
    except Exception as e:
        return _json_error(f"/api/link_diag failed: {e}", 500)


@app.route("/api/controller_diag")
def api_controller_diag():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync.get("ui_window_s", WINDOW_SECONDS_DEFAULT))
        max_points = int(sync.get("ui_ctrl_max_points", MAX_NODE_POINTS_DEFAULT))
        return jsonify(build_controller_timeseries(window_s=window_s, max_points=max_points))
    except Exception as e:
        return _json_error(f"/api/controller_diag failed: {e}", 500)


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
    <div class="sub mono" style="margin-top:0.35rem;" id="scaleLine">scale: …</div>
  </div>

  <div class="grid">
    <div>
      <div class="card">
        <h2 style="font-size:1.05rem;">Topologie</h2>
        <canvas id="meshCanvas"></canvas>
        <div class="sub">Node-Farbe = Ampel. Link-Farbe/Dicke = σ (median im Conv-Window, max beider Richtungen).</div>
      </div>

      <div class="card" style="margin-top:1rem;">
        <h2 style="font-size:1.05rem;">Status (optimal mesh time)</h2>
        <div class="sub">Δ ist <b>Abweichung zur theoretisch optimalen Meshzeit</b> (Konsens = Median von t_mesh).</div>
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
        <h2 style="font-size:1.05rem;">Centered Mesh-Time: ε(node,t) = t_mesh(node) − Konsens (Median) (ms)</h2>
        <div class="sub">Konsens wird pro Bin berechnet. ε ist der Plot, der “wirklich stimmt”.</div>
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
        </div>

        <div class="plots" style="margin-top:1rem;">
          <div>
            <h3 class="small">offset (ms) (legacy)</h3>
            <canvas id="offsetChart" height="150"></canvas>
          </div>
          <div>
            <h3 class="small">err_mesh_vs_wall (ms) (legacy)</h3>
            <canvas id="errChart" height="150"></canvas>
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
            RTT spiky → Queueing/Medium. θ spiky → Asymmetrie/Bootstrap.
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
        Wenn es “stabil aussieht”, aber Ampel ist gelb/rot → schau Reason (stale? correction? stable_links? slew_clipped?).
      </div>
    </div>
  </div>

<script>
  const colors = ['#2ecc71','#3498db','#f1c40f','#e74c3c','#9b59b6','#1abc9c','#e67e22'];

  let meshChart, pairChart, stepChart, stabilityBar, heatmapChart;
  let thetaChart, rttChart, sigmaBar;
  let offsetChart, errChart;
  let deltaDesiredChart, deltaAppliedChart, dtLineChart, slewLineChart;

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
      const v = mapObj[id]?.[field];
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

    offsetChart    = mkLine(document.getElementById('offsetChart').getContext('2d'), 'ms');
    errChart       = mkLine(document.getElementById('errChart').getContext('2d'), 'ms');

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

      const M = ov.mesh || {};
      setMeshHeader(
        M.state || 'YELLOW',
        `${M.state || '…'}: ${M.reason || ''}`,
        `now=${M.now_utc || 'n/a'} · conv_window=${M.conv_window_s || '?'}s · t_mesh_scale=${(M.scale_t_mesh_to_ms ?? '?')} (raw→ms)`
      );

      const T = (M.thresholds || {});
      document.getElementById('thrLine').textContent =
        `fresh≤${T.fresh_s ?? '?'}s  |Δapplied|med≤${T.delta_applied_med_ms ?? '?'}ms  linkσmed≤${T.link_sigma_med_ms ?? '?'}ms  clip≤${Math.round((T.slew_clip_rate ?? 0)*100)}%  K_stable_links=${T.k_stable_links ?? '?'}  warmup≥${T.warmup_min_samples ?? '?'} samples`;

      const off = ov.offenders || {};
      document.getElementById('offendersLine').textContent =
        `worst correction: ${off.worst_node_correction || '—'} · stalest: ${off.stalest_node || '—'} · most clipped: ${off.most_slew_clipped || '—'} · worst link σ: ${off.worst_link_sigma || '—'}`;

      const S = (mesh.meta && mesh.meta.scale) || {};
      document.getElementById('scaleLine').textContent =
        `scale: t_mesh→ms=${S.t_mesh_to_ms ?? '?'}  offset→ms=${S.offset_to_ms ?? '?'}  err_mesh_vs_wall→ms=${S.err_mesh_vs_wall_to_ms ?? '?'}`;

      renderNodeKpis(ov.nodes || []);
      renderStatusTable(ov.nodes || []);

      const nodeStates = {};
      (ov.nodes || []).forEach(n => { nodeStates[n.node_id] = n.state; });

      const meshCanvas = document.getElementById('meshCanvas');
      meshCanvas.width = meshCanvas.clientWidth;
      meshCanvas.height = meshCanvas.clientHeight;
      drawMesh(meshCanvas, topo, nodeStates, ov.link_sigma_med || {});

      updateLine(meshChart, mesh.mesh_series || {}, "mesh_err_ms", "Node ");
      updatePairs(pairChart, mesh.pairs || {});
      updateLine(stepChart, mesh.step_series || {}, "step_ms", "Node ");
      updateBarFromMap(stabilityBar, mesh.stability || {}, "sigma_ms");
      updateHeatmap(heatmapChart, mesh.heatmap || {data:[], n_bins:0});

      updateLine(offsetChart, mesh.offset_series || {}, "offset_ms", "Node ");
      updateLine(errChart, mesh.err_mesh_vs_wall_series || {}, "err_mesh_vs_wall_ms", "Node ");

      const links = link.links || {};
      updateLine(thetaChart, links, "theta_ms", "");
      updateLine(rttChart, links, "rtt_ms", "");
      updateSigmaBar(sigmaBar, link.latest_sigma || {});

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
    setInterval(refresh, 2000);
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
