# -*- coding: utf-8 -*-
# mesh/web_ui.py — CLEAN REWRITE (Backend)
# Python 3.7 compatible

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
# Defaults / UI knobs
# -----------------------------
WINDOW_S_DEFAULT = 600.0
BIN_S_DEFAULT = 0.5
MAX_NODE_POINTS_DEFAULT = 12000
MAX_LINK_POINTS_DEFAULT = 24000
MAX_CTRL_POINTS_DEFAULT = 12000

HEATMAP_MAX_BINS = 60

# Simple convergence rules for overview (you can tune later)
CONV_WINDOW_S = 30.0
MIN_SAMPLES_WARMUP = 8
FRESH_MIN_S = 3.0
FRESH_MULT = 6.0

THRESH_LINK_SIGMA_MED_MS = 2.0

# -----------------------------
# Helpers (robust)
# -----------------------------
def _json_error(msg: str, status: int = 500):
    return make_response(jsonify({"error": str(msg)}), int(status))

def get_conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH))
    c.row_factory = sqlite3.Row
    return c

def _table_cols(conn: sqlite3.Connection, table: str) -> Set[str]:
    try:
        rows = conn.execute("PRAGMA table_info(%s)" % table).fetchall()
        return set([r[1] for r in rows])
    except Exception:
        return set()

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

def median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    if n % 2 == 1:
        return float(xs[n // 2])
    return 0.5 * (float(xs[n // 2 - 1]) + float(xs[n // 2]))

def iqr(xs: List[float]) -> float:
    if len(xs) < 4:
        return 0.0
    xs = sorted(xs)
    q1 = xs[len(xs) // 4]
    q3 = xs[(3 * len(xs)) // 4]
    return float(q3 - q1)

def load_config() -> Dict[str, Any]:
    try:
        if not CFG_PATH.exists():
            return {}
        with CFG_PATH.open("r", encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _beacon_period_s(cfg: Dict[str, Any]) -> float:
    try:
        sync = cfg.get("sync", {}) or {}
        return float(sync.get("beacon_period_s", 0.5))
    except Exception:
        return 0.5

def _fresh_thresh_s(cfg: Dict[str, Any]) -> float:
    return max(float(FRESH_MIN_S), _beacon_period_s(cfg) * float(FRESH_MULT))

def get_topology() -> Dict[str, Any]:
    cfg = load_config()
    nodes = []
    links = []

    if not isinstance(cfg, dict):
        return {"nodes": [], "links": []}

    for node_id, entry in cfg.items():
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
            # undirected link list for drawing
            if node_id < neigh:
                links.append({"source": node_id, "target": neigh})

    return {"nodes": nodes, "links": links}

# -----------------------------
# Unit inference: t_mesh scale
# -----------------------------
def infer_tmesh_to_seconds(rows: List[sqlite3.Row]) -> float:
    """
    Infer factor 'scale' such that:
        t_mesh_seconds = t_mesh_raw * scale

    We estimate r = median(Δt_mesh / Δcreated_at) over consecutive samples (same node).
      r ~ 1        => t_mesh in seconds      => scale ~ 1
      r ~ 1000     => t_mesh in ms          => scale ~ 1/1000
      r ~ 1e6      => t_mesh in microseconds=> scale ~ 1/1e6
    """
    by_node: Dict[str, List[Tuple[float, float]]] = {}
    for r in rows:
        nid = str(r["node_id"]) if "node_id" in r.keys() else ""
        tc = _f(r["created_at"]) if "created_at" in r.keys() else None
        tm = _f(r["t_mesh"]) if "t_mesh" in r.keys() else None
        if nid and tc is not None and tm is not None:
            by_node.setdefault(nid, []).append((tc, tm))

    ratios: List[float] = []
    for pts in by_node.values():
        pts.sort(key=lambda x: x[0])
        for i in range(1, len(pts)):
            dt = float(pts[i][0] - pts[i - 1][0])
            dm = float(pts[i][1] - pts[i - 1][1])
            if dt <= 1e-6:
                continue
            if dm <= 0:
                continue
            ratios.append(dm / dt)

    r_med = median(ratios)
    if r_med is None or r_med <= 0:
        return 1.0

    inv = 1.0 / float(r_med)

    # snap to nice powers to avoid floating jitter
    for snap in (1.0, 1e-3, 1e-6):
        if abs(math.log10(inv) - math.log10(snap)) < 0.25:
            return float(snap)

    return float(inv)

def mesh_offset_ms(created_at_s: float, t_mesh_raw: float, scale_to_seconds: float) -> float:
    return (float(t_mesh_raw) * float(scale_to_seconds) - float(created_at_s)) * 1000.0

# -----------------------------
# DB fetchers (robust to missing cols)
# -----------------------------
def fetch_node_rows(window_s: float, limit: int) -> List[sqlite3.Row]:
    conn = get_conn()
    try:
        cols = _table_cols(conn, "ntp_reference")
        need = {"node_id", "created_at", "t_mesh"}
        if not need.issubset(cols):
            return []

        cutoff = time.time() - float(window_s)

        # node rows: peer_id NULL if column exists, else no filter
        peer_filter = "AND peer_id IS NULL" if "peer_id" in cols else ""

        sel = ["node_id", "created_at", "t_mesh"]
        if "offset" in cols:
            sel.append("offset")
        if "err_mesh_vs_wall" in cols:
            sel.append("err_mesh_vs_wall")

        q = """
            SELECT {sel}
            FROM ntp_reference
            WHERE created_at >= ?
            {peer_filter}
            ORDER BY created_at ASC
            LIMIT ?
        """.format(sel=", ".join(sel), peer_filter=peer_filter)

        return conn.execute(q, (cutoff, int(limit))).fetchall()
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

        cutoff = time.time() - float(window_s)

        sel = ["node_id", "peer_id", "created_at"]
        for c in ("theta_ms", "rtt_ms", "sigma_ms"):
            if c in cols:
                sel.append(c)

        q = """
            SELECT {sel}
            FROM ntp_reference
            WHERE created_at >= ?
              AND peer_id IS NOT NULL
            ORDER BY created_at ASC
            LIMIT ?
        """.format(sel=", ".join(sel))

        return conn.execute(q, (cutoff, int(limit))).fetchall()
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

def fetch_controller_rows(window_s: float, limit: int) -> List[sqlite3.Row]:
    conn = get_conn()
    try:
        cols = _table_cols(conn, "ntp_reference")
        if not {"node_id", "created_at"}.issubset(cols):
            return []

        ctrl_cols = [c for c in ("delta_desired_ms", "delta_applied_ms", "dt_s", "slew_clipped") if c in cols]
        if not ctrl_cols:
            return []

        cutoff = time.time() - float(window_s)
        nonnull = " OR ".join(["%s IS NOT NULL" % c for c in ctrl_cols])

        sel = ["node_id", "created_at"]
        if "peer_id" in cols:
            sel.append("peer_id")
        sel.extend(ctrl_cols)

        q = """
            SELECT {sel}
            FROM ntp_reference
            WHERE created_at >= ?
              AND ({nonnull})
            ORDER BY created_at ASC
            LIMIT ?
        """.format(sel=", ".join(sel), nonnull=nonnull)

        return conn.execute(q, (cutoff, int(limit))).fetchall()
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass

def controller_delta_scale_to_ms(rows: List[sqlite3.Row]) -> float:
    """
    Deine delta_*_ms Werte sehen nach Sekunden aus (z.B. 0.03 ~= 30ms).
    Heuristik: wenn median(|delta|) < 1.0 -> treat as seconds, scale 1000.
    """
    vals: List[float] = []
    for r in rows:
        for k in ("delta_desired_ms", "delta_applied_ms"):
            if k in r.keys():
                v = _f(r[k])
                if v is not None:
                    vals.append(abs(float(v)))
    m = median(vals)
    if m is None:
        return 1.0
    return 1000.0 if float(m) < 1.0 else 1.0

# -----------------------------
# Mesh diagnostics (stable, matches JS)
# -----------------------------
def build_mesh_diag(window_s, bin_s, max_points):
    rows = fetch_node_rows(window_s, max_points)
    if not rows:
        return {
            "meta": {"note": "no data"},
            "mesh_series_ffill": {}, "step_series_ffill": {}, "pairs_ffill": {},
            "mesh_series_window": {}, "step_series_window": {}, "pairs_window": {},
            "mesh_series_event": {}, "step_series_event": {}, "pairs_event": {},
        }

    if bin_s <= 0:
        bin_s = BIN_S_DEFAULT

    scale = infer_tmesh_to_seconds(rows)

    # ---- config-driven TTL/W defaults (safe fallbacks) ----
    # TTL: how long we keep last value for ffill/event before marking stale
    # W:   sliding window half-width for nearest-sample selection
    cfg = {}
    try:
        cfg = json.load(open(str(CFG_PATH))) if CFG_PATH.exists() else {}
    except Exception:
        cfg = {}
    sync = cfg.get("sync", {}) if isinstance(cfg, dict) else {}
    beacon_period_s = 0.5
    try:
        beacon_period_s = float(sync.get("beacon_period_s", 0.5))
    except Exception:
        beacon_period_s = 0.5

    ttl_s = sync.get("ui_ttl_s", None)
    if ttl_s is None:
        ttl_s = max(2.0, 3.0 * beacon_period_s, 4.0 * float(bin_s))
    try:
        ttl_s = float(ttl_s)
    except Exception:
        ttl_s = max(2.0, 3.0 * beacon_period_s, 4.0 * float(bin_s))

    win_s = sync.get("ui_win_s", None)
    if win_s is None:
        win_s = max(1.0 * beacon_period_s, 2.0 * float(bin_s))
    try:
        win_s = float(win_s)
    except Exception:
        win_s = max(1.0 * beacon_period_s, 2.0 * float(bin_s))

    # ---- prepare samples per node: [(t, mesh_offset_ms)] ----
    by_node = {}
    t_min = None
    t_max = None

    for r in rows:
        n = r["node_id"]
        t = _f(r["created_at"])
        tm = _f(r["t_mesh"])
        if not n or t is None or tm is None:
            continue
        mo = mesh_offset_ms(t, tm, scale)
        by_node.setdefault(n, []).append((t, mo))
        if t_min is None or t < t_min:
            t_min = t
        if t_max is None or t > t_max:
            t_max = t

    if not by_node or t_min is None or t_max is None:
        return {
            "meta": {"note": "no valid samples"},
            "mesh_series_ffill": {}, "step_series_ffill": {}, "pairs_ffill": {},
            "mesh_series_window": {}, "step_series_window": {}, "pairs_window": {},
            "mesh_series_event": {}, "step_series_event": {}, "pairs_event": {},
        }

    nodes = sorted(by_node.keys())
    for n in nodes:
        by_node[n].sort(key=lambda x: x[0])

    # helper: compute step-series from mesh-series
    def _make_steps(mesh_series):
        out = {}
        for n, pts in mesh_series.items():
            pts_sorted = sorted(pts, key=lambda p: p["t_wall"])
            prev = None
            s = []
            for p in pts_sorted:
                cur = p["mesh_err_ms"]
                if prev is None:
                    d = 0.0
                else:
                    d = float(cur - prev)
                s.append({"t_wall": float(p["t_wall"]), "step_ms": float(d)})
                prev = cur
            out[n] = s
        return out

    # helper: compute pairs from per-bin eps map
    def _pairs_from_eps_timeseries(eps_by_time):
        pairs = {}
        for t_wall, eps in eps_by_time:
            ns = sorted(eps.keys())
            for i in range(len(ns)):
                for j in range(i + 1, len(ns)):
                    pid = "%s-%s" % (ns[i], ns[j])
                    pairs.setdefault(pid, []).append({
                        "t_wall": float(t_wall),
                        "delta_ms": float(eps[ns[i]] - eps[ns[j]]),
                    })
        return pairs

    # =========================================================================
    # (A) BINNED + FORWARD-FILL (TTL)
    # =========================================================================
    idx0 = int(math.floor(float(t_min) / float(bin_s)))
    idx1 = int(math.floor(float(t_max) / float(bin_s)))

    # pointers per node into their sample list
    ptr = dict((n, 0) for n in nodes)
    last_val = {}      # node -> last mo
    last_time = {}     # node -> last sample time

    mesh_ffill = dict((n, []) for n in nodes)
    eps_times_ffill = []  # list[(t_wall, eps_map)]

    for idx in range(idx0, idx1 + 1):
        t_center = (float(idx) + 0.5) * float(bin_s)

        # advance each node pointer up to t_center (take latest <= t_center)
        for n in nodes:
            pts = by_node[n]
            p = ptr[n]
            while p < len(pts) and pts[p][0] <= t_center:
                last_time[n] = pts[p][0]
                last_val[n] = pts[p][1]
                p += 1
            ptr[n] = p

        # collect fresh values within TTL
        fresh = {}
        for n in nodes:
            lt = last_time.get(n)
            lv = last_val.get(n)
            if lt is None or lv is None:
                continue
            if (t_center - float(lt)) <= float(ttl_s):
                fresh[n] = float(lv)

        if len(fresh) < 2:
            continue

        cons = median(list(fresh.values()))
        if cons is None:
            continue

        eps = {}
        for n, v in fresh.items():
            e = float(v - cons)
            eps[n] = e
            mesh_ffill[n].append({"t_wall": float(t_center), "mesh_err_ms": float(e)})

        eps_times_ffill.append((t_center, eps))

    # drop empty series
    mesh_series_ffill = dict((n, pts) for (n, pts) in mesh_ffill.items() if pts)
    step_series_ffill = _make_steps(mesh_series_ffill)
    pairs_ffill = _pairs_from_eps_timeseries(eps_times_ffill)

    # =========================================================================
    # (B) BINNED + SLIDING WINDOW (nearest sample within ±win_s)
    # =========================================================================
    # Build fast pointers for nearest search via forward scan.
    ptr2 = dict((n, 0) for n in nodes)
    mesh_window = dict((n, []) for n in nodes)
    eps_times_window = []

    for idx in range(idx0, idx1 + 1):
        t_center = (float(idx) + 0.5) * float(bin_s)
        eps_candidates = {}

        for n in nodes:
            pts = by_node[n]
            p = ptr2[n]

            # move pointer while next sample time < t_center (so p is near center)
            while p + 1 < len(pts) and pts[p + 1][0] < t_center:
                p += 1
            ptr2[n] = p

            # check p and p+1 as nearest candidates
            best = None
            best_dt = None
            for q in (p, p + 1):
                if q < 0 or q >= len(pts):
                    continue
                dt = abs(float(pts[q][0]) - float(t_center))
                if best_dt is None or dt < best_dt:
                    best_dt = dt
                    best = pts[q][1]

            if best is None or best_dt is None:
                continue
            if float(best_dt) <= float(win_s):
                eps_candidates[n] = float(best)

        if len(eps_candidates) < 2:
            continue

        cons = median(list(eps_candidates.values()))
        if cons is None:
            continue

        eps = {}
        for n, v in eps_candidates.items():
            e = float(v - cons)
            eps[n] = e
            mesh_window[n].append({"t_wall": float(t_center), "mesh_err_ms": float(e)})

        eps_times_window.append((t_center, eps))

    mesh_series_window = dict((n, pts) for (n, pts) in mesh_window.items() if pts)
    step_series_window = _make_steps(mesh_series_window)
    pairs_window = _pairs_from_eps_timeseries(eps_times_window)

    # =========================================================================
    # (C) EVENT-BASED (controller-like): every new sample updates consensus
    # =========================================================================
    # Flatten all events: (t, node, mo)
    events = []
    for n in nodes:
        for (t, mo) in by_node[n]:
            events.append((float(t), n, float(mo)))
    events.sort(key=lambda x: x[0])

    last_val_e = {}
    last_time_e = {}
    mesh_event = dict((n, []) for n in nodes)
    eps_times_event = []

    for (t, n, mo) in events:
        last_val_e[n] = float(mo)
        last_time_e[n] = float(t)

        fresh = {}
        for nn in nodes:
            lt = last_time_e.get(nn)
            lv = last_val_e.get(nn)
            if lt is None or lv is None:
                continue
            if (float(t) - float(lt)) <= float(ttl_s):
                fresh[nn] = float(lv)

        if len(fresh) < 2:
            continue

        cons = median(list(fresh.values()))
        if cons is None:
            continue

        # We only append a point for the node that produced the event (controller-style).
        e = float(last_val_e[n] - cons)
        mesh_event[n].append({"t_wall": float(t), "mesh_err_ms": float(e)})

        # For pairs you need full eps map at this time:
        eps_map = dict((nn, float(v - cons)) for (nn, v) in fresh.items())
        eps_times_event.append((t, eps_map))

    mesh_series_event = dict((n, pts) for (n, pts) in mesh_event.items() if pts)
    step_series_event = _make_steps(mesh_series_event)
    pairs_event = _pairs_from_eps_timeseries(eps_times_event)

    return {
        "meta": {
            "window_s": float(window_s),
            "bin_s": float(bin_s),
            "ttl_s": float(ttl_s),
            "win_s": float(win_s),
            "t_mesh_to_seconds": float(scale),
            "nodes": nodes,
        },
        "mesh_series_ffill": mesh_series_ffill,
        "step_series_ffill": step_series_ffill,
        "pairs_ffill": pairs_ffill,

        "mesh_series_window": mesh_series_window,
        "step_series_window": step_series_window,
        "pairs_window": pairs_window,

        "mesh_series_event": mesh_series_event,
        "step_series_event": step_series_event,
        "pairs_event": pairs_event,
    }


# -----------------------------
# Link diagnostics (stable, matches JS)
# -----------------------------
def build_link_diag(window_s: float, bin_s: float, max_points: int) -> Dict[str, Any]:
    rows = fetch_link_rows(window_s, max_points)
    if not rows or bin_s <= 0:
        return {"links": {}, "latest_sigma": {}, "meta": {"window_s": float(window_s), "bin_s": float(bin_s), "x_axis": "created_at"}}

    # bins[idx][lid] -> lists
    bins: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    for r in rows:
        t = _f(r["created_at"])
        if t is None:
            continue
        idx = int(math.floor(float(t) / float(bin_s)))
        lid = "%s->%s" % (str(r["node_id"]), str(r["peer_id"]))
        d = bins.setdefault(idx, {}).setdefault(lid, {"theta_ms": [], "rtt_ms": [], "sigma_ms": []})

        for k in ("theta_ms", "rtt_ms", "sigma_ms"):
            if k in r.keys():
                v = _f(r[k])
                if v is not None:
                    d[k].append(float(v))

    out: Dict[str, List[Dict[str, float]]] = {}
    latest_sigma: Dict[str, float] = {}

    for idx in sorted(bins.keys()):
        t_bin = (float(idx) + 0.5) * float(bin_s)
        for lid, d in bins[idx].items():
            obj: Dict[str, float] = {"t_wall": float(t_bin)}
            m = median(d["theta_ms"])
            if m is not None:
                obj["theta_ms"] = float(m)
            m = median(d["rtt_ms"])
            if m is not None:
                obj["rtt_ms"] = float(m)
            m = median(d["sigma_ms"])
            if m is not None:
                obj["sigma_ms"] = float(m)
                latest_sigma[lid] = float(m)
            out.setdefault(lid, []).append(obj)

    return {"links": out, "latest_sigma": latest_sigma, "meta": {"window_s": float(window_s), "bin_s": float(bin_s), "x_axis": "created_at"}}

# -----------------------------
# Controller diagnostics (matches JS)
# -----------------------------
def build_controller_diag(window_s: float, max_points: int) -> Dict[str, Any]:
    rows = fetch_controller_rows(window_s, max_points)
    if not rows:
        return {"controller": {}, "meta": {"window_s": float(window_s), "x_axis": "created_at", "delta_scale_to_ms": 1.0}}

    scale = controller_delta_scale_to_ms(rows)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        nid = str(r["node_id"])
        t = _f(r["created_at"])
        if not nid or t is None:
            continue

        obj: Dict[str, Any] = {"t_wall": float(t)}

        if "delta_desired_ms" in r.keys():
            v = _f(r["delta_desired_ms"])
            if v is not None:
                obj["delta_desired_ms"] = float(v) * float(scale)

        if "delta_applied_ms" in r.keys():
            v = _f(r["delta_applied_ms"])
            if v is not None:
                obj["delta_applied_ms"] = float(v) * float(scale)

        if "dt_s" in r.keys():
            v = _f(r["dt_s"])
            if v is not None:
                obj["dt_s"] = float(v)

        if "slew_clipped" in r.keys() and r["slew_clipped"] is not None:
            try:
                obj["slew_clipped"] = 1 if int(r["slew_clipped"]) else 0
            except Exception:
                pass

        out.setdefault(nid, []).append(obj)

    for nid in out.keys():
        out[nid].sort(key=lambda p: p["t_wall"])

    return {"controller": out, "meta": {"window_s": float(window_s), "x_axis": "created_at", "delta_scale_to_ms": float(scale)}}

# -----------------------------
# Overview (minimal but useful, matches your JS needs)
# -----------------------------
def compute_overview(window_s: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    now = time.time()
    eff = max(float(window_s), float(CONV_WINDOW_S))
    node_rows = fetch_node_rows(eff, MAX_NODE_POINTS_DEFAULT)
    link_rows = fetch_link_rows(eff, MAX_LINK_POINTS_DEFAULT)

    fresh_thr = _fresh_thresh_s(cfg)
    conv_cut = float(now) - float(CONV_WINDOW_S)

    # latest per node
    latest: Dict[str, sqlite3.Row] = {}
    by_node: Dict[str, List[sqlite3.Row]] = {}
    for r in node_rows:
        nid = str(r["node_id"])
        by_node.setdefault(nid, []).append(r)
        latest[nid] = r

    scale = infer_tmesh_to_seconds(list(latest.values())) if latest else 1.0

    # consensus now from latest mesh_offset_ms
    offs_now: List[float] = []
    offs_map: Dict[str, float] = {}
    for nid, r in latest.items():
        t = _f(r["created_at"])
        tm = _f(r["t_mesh"])
        if t is None or tm is None:
            continue
        mo = mesh_offset_ms(float(t), float(tm), float(scale))
        offs_now.append(mo)
        offs_map[nid] = mo
    cons_now = median(offs_now)

    # link sigma med in conv window (directed)
    sigs_by_link: Dict[str, List[float]] = {}
    last_seen_link: Dict[str, float] = {}
    for r in link_rows:
        t = _f(r["created_at"])
        if t is None:
            continue
        lid = "%s->%s" % (str(r["node_id"]), str(r["peer_id"]))
        last_seen_link[lid] = max(last_seen_link.get(lid, 0.0), float(t))
        if float(t) < conv_cut:
            continue
        if "sigma_ms" in r.keys():
            sg = _f(r["sigma_ms"])
            if sg is not None:
                sigs_by_link.setdefault(lid, []).append(float(sg))

    link_sigma_med: Dict[str, float] = {}
    for lid, vals in sigs_by_link.items():
        m = median(vals)
        if m is not None:
            link_sigma_med[lid] = float(m)

    # node states (very simple: warmup/fresh + outgoing stable link)
    nodes_out: List[Dict[str, Any]] = []
    for nid in sorted(latest.keys()):
        r_last = latest[nid]
        t_last = _f(r_last["created_at"])
        age_s = float(now - t_last) if t_last is not None else None
        fresh_ok = (age_s is not None and float(age_s) <= float(fresh_thr))

        recent = []
        for rr in by_node.get(nid, []):
            t = _f(rr["created_at"])
            if t is not None and float(t) >= conv_cut:
                recent.append(rr)

        enough = len(recent) >= int(MIN_SAMPLES_WARMUP)

        mesh_err_now_ms = None
        if cons_now is not None and nid in offs_map:
            mesh_err_now_ms = float(offs_map[nid] - float(cons_now))

        # count stable outgoing links
        stable_links = 0
        total_considered = 0
        for lid, sig in link_sigma_med.items():
            if not lid.startswith(nid + "->"):
                continue
            ls = last_seen_link.get(lid)
            if ls is None:
                continue
            if float(now - ls) > float(fresh_thr):
                continue
            total_considered += 1
            if float(sig) <= float(THRESH_LINK_SIGMA_MED_MS):
                stable_links += 1

        if not enough:
            state, reason = "YELLOW", "warming up"
        elif not fresh_ok:
            state, reason = "RED", "stale data (age %.1fs)" % (age_s if age_s is not None else -1.0)
        else:
            # if no links known, keep yellow; else green if >=1 stable
            if total_considered == 0:
                state, reason = "YELLOW", "no fresh link sigma"
            elif stable_links >= 1:
                state, reason = "GREEN", "converged"
            else:
                state, reason = "YELLOW", "unstable links"

        nodes_out.append({
            "node_id": nid,
            "state": state,
            "reason": reason,
            "last_seen_utc": _utc(t_last),
            "age_s": age_s,
            "mesh_err_now_ms": mesh_err_now_ms,
            "mesh_rate_ms_s": None,  # keep simple (optional)
            "med_abs_delta_applied_ms": None,
            "slew_clip_rate": None,
            "stable_links": stable_links,
            "k_stable_links": 1,
            "links_considered": total_considered,
        })

    mesh_state = "GREEN" if nodes_out and all(n["state"] == "GREEN" for n in nodes_out) else "YELLOW"
    if any(n["state"] == "RED" for n in nodes_out):
        mesh_state = "RED"

    return {
        "mesh": {
            "state": mesh_state,
            "reason": "converged" if mesh_state == "GREEN" else "not converged",
            "now_utc": _utc(now),
            "conv_window_s": float(CONV_WINDOW_S),
            "consensus_now_offset_ms": cons_now,
            "t_mesh_to_seconds": float(scale),
            "thresholds": {
                "fresh_s": float(fresh_thr),
                "link_sigma_med_ms": float(THRESH_LINK_SIGMA_MED_MS),
                "warmup_min_samples": int(MIN_SAMPLES_WARMUP),
                "delta_applied_med_ms": 0.5,
                "slew_clip_rate": 0.2,
                "k_stable_links": 1,
            },
        },
        "nodes": nodes_out,
        "offenders": {
            "worst_node_correction": None,
            "stalest_node": None,
            "worst_link_sigma": None,
            "most_slew_clipped": None,
        },
        "link_sigma_med": link_sigma_med,
        "link_last_seen": last_seen_link,
    }

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def index():
    # If you keep your TEMPLATE below, this renders it; otherwise swap to your file/other renderer.
    topo = get_topology()
    return render_template_string(TEMPLATE, topo=topo, db_path=str(DB_PATH))

@app.route("/api/topology")
def api_topology():
    try:
        return jsonify(get_topology())
    except Exception as e:
        return _json_error("/api/topology failed: %s" % e, 500)

@app.route("/api/overview")
def api_overview():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync.get("ui_window_s", WINDOW_S_DEFAULT))
        return jsonify(compute_overview(window_s, cfg))
    except Exception as e:
        return _json_error("/api/overview failed: %s" % e, 500)

@app.route("/api/mesh_diag")
def api_mesh_diag():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync.get("ui_window_s", WINDOW_S_DEFAULT))
        bin_s = float(sync.get("ui_bin_s", BIN_S_DEFAULT))
        max_points = int(sync.get("ui_max_points", MAX_NODE_POINTS_DEFAULT))
        return jsonify(build_mesh_diag(window_s, bin_s, max_points))
    except Exception as e:
        return _json_error("/api/mesh_diag failed: %s" % e, 500)

@app.route("/api/link_diag")
def api_link_diag():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync.get("ui_window_s", WINDOW_S_DEFAULT))
        bin_s = float(sync.get("ui_bin_s", BIN_S_DEFAULT))
        max_points = int(sync.get("ui_link_max_points", MAX_LINK_POINTS_DEFAULT))
        return jsonify(build_link_diag(window_s, bin_s, max_points))
    except Exception as e:
        return _json_error("/api/link_diag failed: %s" % e, 500)

@app.route("/api/controller_diag")
def api_controller_diag():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync.get("ui_window_s", WINDOW_S_DEFAULT))
        max_points = int(sync.get("ui_ctrl_max_points", MAX_CTRL_POINTS_DEFAULT))
        return jsonify(build_controller_diag(window_s, max_points))
    except Exception as e:
        return _json_error("/api/controller_diag failed: %s" % e, 500)

# -----------------------------
# Boilerplate
# -----------------------------
def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Storage(str(DB_PATH))


# -----------------------------
# Template (Single Page) - same as before
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
          <span class="small muted">Mesh mode:</span>
          <select id="meshMode">
            <option value="event" selected>event (controller-like)</option>
            <option value="ffill">ffill (bin + TTL)</option>
            <option value="window">window (bin + nearest)</option>
          </select>
        
          <span class="small muted" style="margin-left:0.5rem;">Controller node:</span>
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
          Hier ist "Meshzeit" bewusst als <span class="mono">(t_mesh - created_at)</span> definiert (Offset-artig), damit Sampling-Zeitverschiebungen nicht als Fehler erscheinen.
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
          Das ist der Plot, der "wirklich stimmt".
        </div>
        <div class="row small muted" style="margin-bottom:0.4rem;">
          <span>Mesh-Plot-Modus:</span>
          <select id="meshMode">
            <option value="strict">Strict (≥2 Nodes / Bin)</option>
            <option value="carry">Carry-Forward (stabil)</option>
            <option value="raw">Raw (alle Punkte)</option>
          </select>
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
        Reality check: Wenn es "stabil aussieht", aber Ampel ist gelb/rot → schau Reason: stale? correction? stable_links? slew_clipped?
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
  
  function carryForwardMap(seriesMap, field){
    // seriesMap: { id: [{t:..., <field>:...}, ...] }
    // füllt fehlende Werte pro Serie mit letztem bekannten Wert (Last-Value Hold)
    const out = {};
    Object.keys(seriesMap || {}).forEach(id=>{
      const pts = seriesMap[id] || [];
      let last = null;
      const filled = [];
      for(const p of pts){
        if(!p) continue;
        const v = p[field];
        if(v !== null && v !== undefined && Number.isFinite(v)){
          last = v;
          filled.push(p);
        } else if(last !== null){
          const q = Object.assign({}, p);
          q[field] = last;
          filled.push(q);
        }
      }
      out[id] = filled;
    });
    return out;
  }


  function updateLine(chart, seriesMap, field, labelPrefix=''){
    const modeEl = document.getElementById('meshMode');
    const mode = modeEl ? modeEl.value : 'strict';

    let dataMap = seriesMap || {};

    // carry-forward nur für Mesh/Step/Link Serien sinnvoll
    if(mode === 'carry'){
      dataMap = carryForwardMap(dataMap, field);
    }

    const ids = Object.keys(dataMap || {}).sort();
    chart.data.datasets = [];

    ids.forEach((id, idx)=>{
      const c = colors[idx % colors.length];

      const pts = (dataMap[id] || [])
        .filter(p => p && p[field] !== null && p[field] !== undefined && Number.isFinite(p[field]))
        // WICHTIG: mesh_diag liefert t in SEKUNDEN (nicht t_wall!)
        .map(p => ({ x: new Date(p.t * 1000), y: p[field] }));

      chart.data.datasets.push({
        label: `${labelPrefix}${id}`,
        data: pts,
        borderColor: c,
        borderWidth: 1.6
      });
   });

    chart.update();
  }


  function updatePairs(chart, pairs){
    const modeEl = document.getElementById('meshMode');
    const mode = modeEl ? modeEl.value : 'strict';

    const ids = Object.keys(pairs || {}).sort();
    chart.data.datasets = [];

    ids.forEach((id, idx)=>{
      const c = colors[idx % colors.length];

      let pts = (pairs[id] || [])
        .filter(p => p && p.y !== null && p.y !== undefined && Number.isFinite(p.y))
        .map(p => ({ x: new Date(p.t * 1000), y: p.y }));

      // carry-forward für pairs (optional, macht Sinn für "stabil")
      if(mode === 'carry' && pts.length){
        let last = null;
          pts = pts.map(p=>{
          if(Number.isFinite(p.y)){ last = p.y; return p; }
          if(last !== null) return { x: p.x, y: last };
         return null;
        }).filter(Boolean);
      }

      chart.data.datasets.push({
        label: id,
        data: pts,
        borderColor: c,
        borderWidth: 1.6
      });
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
  
  function pickMeshMode(mesh){
  const mode = (document.getElementById('meshMode')?.value) || 'event';
  const ms = mesh[`mesh_series_${mode}`] || {};
  const ss = mesh[`step_series_${mode}`] || {};
  const ps = mesh[`pairs_${mode}`] || {};
  return { mode, ms, ss, ps };
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

      // plots (mesh mode selectable)
      const picked = pickMeshMode(mesh);
      updateLine(meshChart, mesh.mesh_series || {}, "y", "Node ");
      updatePairs(pairChart, picked.ps);
      updateLine(stepChart, mesh.step_series || {}, "y", "Node ");

    
      // optional: falls du stability/heatmap noch nicht auf die neuen Modi umgebaut hast,
      // lass sie erstmal leer, statt "nix zeichnet" Bugs zu erzeugen.
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
    document.getElementById('meshMode').addEventListener('change', ()=>refresh());
    applyDebugVisibility();
    refresh();
    setInterval(refresh, 1500);
  });
</script>

</body>
</html>
"""


if __name__ == "__main__":
    ensure_db()
    print("Starting MeshTime Dashboard on http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)