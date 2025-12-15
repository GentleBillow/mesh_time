# -*- coding: utf-8 -*-
# mesh/web_ui.py
#
# MeshTime Dashboard — Single Page (Operator + Debug)
#
# PRINCIPLES:
# - 10s answer: "Is mesh synchronized? If not, why (which node/link)?"
# - Mesh-Time first-class: show t_mesh(node) relative to consensus (median) over time
# - Deltas are diagnosis (links, controller), not truth
# - created_at (sink clock) is the ONLY x-axis
# - Units: seconds vs ms are explicit; convert exactly once
# - Must never crash on None or missing columns

from __future__ import annotations

import json
import math
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flask import Flask, jsonify, make_response, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "mesh_data.sqlite"
CFG_PATH = BASE_DIR / "config" / "nodes.json"

from mesh.storage import Storage  # ensure schema

app = Flask(__name__)

# -----------------------------
# Defaults / thresholds
# -----------------------------
WINDOW_S_DEFAULT = 120.0
BIN_S_DEFAULT = 0.5
MAX_NODE_POINTS = 12000
MAX_LINK_POINTS = 24000

# convergence / operator thresholds
CONV_WINDOW_S = 30.0
MIN_SAMPLES_WARMUP = 8

# Freshness: default is max(T_FRESH_MIN, beacon_period * mult)
T_FRESH_MIN_S = 3.0
T_FRESH_MULT = 6.0  # 2*period*3 from your text => 6x

T_DELTA_APPLIED_MED_MS = 0.5
T_LINK_SIGMA_MED_MS = 2.0
T_SLEW_CLIP_RATE = 0.20  # 20%

HEATMAP_MAX_BINS = 50


# ------------------------------------------------------------
# Helpers (robust, never crash)
# ------------------------------------------------------------
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


def _i(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _utc(ts: Optional[float]) -> str:
    if ts is None:
        return "n/a"
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if (n % 2) else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _quantile_sorted(xs: List[float], q: float) -> float:
    n = len(xs)
    if n <= 1:
        return float(xs[0]) if xs else 0.0
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


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
                links.append({"a": node_id, "b": neigh})

    return {"nodes": nodes, "links": links}


# ------------------------------------------------------------
# DB readers (robust to missing columns)
# ------------------------------------------------------------
def fetch_node_rows(window_s: float, limit: int) -> List[sqlite3.Row]:
    conn = get_conn()
    try:
        cols = _table_cols(conn, "ntp_reference")
        if not {"node_id", "created_at"}.issubset(cols):
            return []

        wanted = [
            "id", "created_at", "node_id",
            # time / mesh
            "t_mesh", "offset",
            # controller debug
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

        wanted = ["id", "created_at", "node_id", "peer_id", "theta_ms", "rtt_ms", "sigma_ms"]
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


# ------------------------------------------------------------
# Core aggregation
# ------------------------------------------------------------
def _infer_beacon_period(cfg: Dict[str, Any]) -> float:
    try:
        sync_cfg = cfg.get("sync", {}) or {}
        return float(sync_cfg.get("beacon_period_s", 0.5))
    except Exception:
        return 0.5


def _fresh_thresh_s(cfg: Dict[str, Any]) -> float:
    bp = _infer_beacon_period(cfg)
    return max(T_FRESH_MIN_S, float(bp) * float(T_FRESH_MULT))


def build_node_timeseries_mesh(window_s: float, bin_s: float) -> Dict[str, List[Dict[str, float]]]:
    """
    Returns per node: [{t, mesh_ms}] where mesh_ms = (t_mesh_ms - consensus_ms) per bin.
    t_mesh assumed seconds (or any unit) — we convert to ms ONCE if it looks like seconds.

    Heuristic:
      - if typical |t_mesh| is < 1e6 -> assume seconds and multiply by 1000
      - if it's already ms-like (e.g., 1e12?), keep as-is? (unlikely)
    """
    rows = fetch_node_rows(window_s, MAX_NODE_POINTS)
    if not rows:
        return {}

    # collect times
    ts = [_f(row_get(r, "created_at")) for r in rows]
    ts = [t for t in ts if t is not None]
    if not ts:
        return {}
    t0 = min(ts)

    # gather raw t_mesh magnitudes for heuristic
    tmesh_vals = []
    for r in rows:
        tm = _f(row_get(r, "t_mesh"))
        if tm is not None:
            tmesh_vals.append(abs(tm))
    scale = 1000.0  # default seconds->ms
    if tmesh_vals:
        med = _median(tmesh_vals) or 0.0
        # if it's already in ms (e.g. values ~1e3..1e7 could still be seconds-of-epoch? but t_mesh likely small)
        # treat "epoch seconds" (1.7e9) also as seconds -> ms (fine)
        # treat "ms epoch" (1.7e12) -> do NOT scale
        if med > 1e11:
            scale = 1.0

    bins: Dict[int, Dict[str, Tuple[float, float]]] = {}  # idx -> node -> (t_last, tmesh_ms)
    for r in rows:
        node = str(row_get(r, "node_id", "") or "")
        t = _f(row_get(r, "created_at"))
        tm = _f(row_get(r, "t_mesh"))
        if not node or t is None or tm is None:
            continue
        idx = int((t - t0) / max(bin_s, 1e-6))
        bucket = bins.setdefault(idx, {})
        prev = bucket.get(node)
        if (prev is None) or (t >= prev[0]):
            bucket[node] = (t, tm * scale)

    out: Dict[str, List[Dict[str, float]]] = {}
    for idx in sorted(bins.keys()):
        bucket = bins[idx]
        if len(bucket) < 2:
            continue
        consensus = _median([v for (_t, v) in bucket.values()])
        if consensus is None:
            continue
        t_bin = t0 + (idx + 0.5) * bin_s
        for node, (_t, tmesh_ms) in bucket.items():
            out.setdefault(node, []).append({"t": float(t_bin), "mesh_ms": float(tmesh_ms - consensus)})

    # sort
    for node, pts in out.items():
        pts.sort(key=lambda p: p["t"])
    return out


def build_controller_timeseries(window_s: float) -> Dict[str, List[Dict[str, Any]]]:
    rows = fetch_node_rows(window_s, MAX_NODE_POINTS)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        node = str(row_get(r, "node_id", "") or "")
        t = _f(row_get(r, "created_at"))
        if not node or t is None:
            continue
        obj: Dict[str, Any] = {"t": float(t)}

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

        out.setdefault(node, []).append(obj)

    for node, pts in out.items():
        pts.sort(key=lambda p: p["t"])
    return out


def build_links_table_and_series(window_s: float, bin_s: float) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    returns:
      - table: per directed link medians + age + UP/DOWN
      - series: binned timeseries per link for heatmap (metric values)
    """
    rows = fetch_link_rows(window_s, MAX_LINK_POINTS)
    now = time.time()
    if not rows:
        return [], {}

    # bin per link
    ts = [_f(row_get(r, "created_at")) for r in rows]
    ts = [t for t in ts if t is not None]
    if not ts:
        return [], {}
    t0 = min(ts)

    # raw lists for medians (conv window)
    conv_cut = now - CONV_WINDOW_S
    vals_conv: Dict[str, Dict[str, List[float]]] = {}
    last_seen: Dict[str, float] = {}

    # binned series for heatmap
    bins: Dict[int, Dict[str, Dict[str, float]]] = {}  # idx -> link -> {theta,rtt,sigma}

    for r in rows:
        t = _f(row_get(r, "created_at"))
        nid = str(row_get(r, "node_id", "") or "")
        pid = str(row_get(r, "peer_id", "") or "")
        if t is None or not nid or not pid:
            continue
        lid = f"{nid}→{pid}"
        last_seen[lid] = max(last_seen.get(lid, 0.0), float(t))

        theta = _f(row_get(r, "theta_ms"))
        rtt = _f(row_get(r, "rtt_ms"))
        sigma = _f(row_get(r, "sigma_ms"))

        if t >= conv_cut:
            box = vals_conv.setdefault(lid, {"theta": [], "rtt": [], "sigma": []})
            if theta is not None:
                box["theta"].append(theta)
            if rtt is not None:
                box["rtt"].append(rtt)
            if sigma is not None:
                box["sigma"].append(sigma)

        # heatmap bins (full window)
        idx = int((t - t0) / max(bin_s, 1e-6))
        b = bins.setdefault(idx, {})
        prev = b.get(lid)
        # keep the latest in the bin
        if prev is None or t >= prev.get("_t", -1):
            obj: Dict[str, float] = {"_t": float(t)}
            if theta is not None:
                obj["theta_ms"] = float(theta)
            if rtt is not None:
                obj["rtt_ms"] = float(rtt)
            if sigma is not None:
                obj["sigma_ms"] = float(sigma)
            b[lid] = obj

    # table build
    table: List[Dict[str, Any]] = []
    for lid in sorted(set(list(last_seen.keys()) + list(vals_conv.keys()))):
        ls = last_seen.get(lid)
        age = (now - ls) if ls is not None else None
        box = vals_conv.get(lid, {"theta": [], "rtt": [], "sigma": []})
        theta_med = _median(box["theta"])
        rtt_med = _median(box["rtt"])
        sigma_med = _median(box["sigma"])
        state = "UP" if (age is not None and age <= 2.0 * window_s) else "DOWN"

        table.append({
            "link": lid,
            "state": state,
            "age_s": age,
            "theta_med_ms": theta_med,
            "rtt_med_ms": rtt_med,
            "sigma_med_ms": sigma_med,
            "n_sigma": len(box["sigma"]),
        })

    # series for heatmap: downsample bins to max bins
    idxs = sorted(bins.keys())
    if not idxs:
        return table, {}

    idx_min, idx_max = idxs[0], idxs[-1]
    total_bins = max(1, idx_max - idx_min + 1)
    display_bins = min(HEATMAP_MAX_BINS, total_bins)

    series: Dict[str, List[Dict[str, Any]]] = {}
    if display_bins == total_bins:
        for idx in idxs:
            t_bin = t0 + (idx + 0.5) * bin_s
            for lid, obj in bins[idx].items():
                rec = {"t": float(t_bin)}
                for k in ["theta_ms", "rtt_ms", "sigma_ms"]:
                    if k in obj:
                        rec[k] = obj[k]
                series.setdefault(lid, []).append(rec)
    else:
        # compress bins into display_bins (mean within compressed bin)
        accum: Dict[Tuple[str, int], Dict[str, List[float]]] = {}
        for idx in idxs:
            rel = (idx - idx_min) / max(1.0, float(total_bins))
            d_idx = int(rel * display_bins)
            if d_idx >= display_bins:
                d_idx = display_bins - 1
            for lid, obj in bins[idx].items():
                key = (lid, d_idx)
                a = accum.setdefault(key, {"theta_ms": [], "rtt_ms": [], "sigma_ms": []})
                for k in ["theta_ms", "rtt_ms", "sigma_ms"]:
                    if k in obj:
                        a[k].append(float(obj[k]))

        for (lid, d_idx), a in accum.items():
            idx_center = idx_min + (d_idx + 0.5) * (total_bins / display_bins)
            t_center = t0 + (idx_center + 0.5) * bin_s
            rec: Dict[str, Any] = {"t": float(t_center)}
            for k in ["theta_ms", "rtt_ms", "sigma_ms"]:
                rec[k] = float(sum(a[k]) / len(a[k])) if a[k] else None
            series.setdefault(lid, []).append(rec)

    for lid, pts in series.items():
        pts.sort(key=lambda p: p["t"])

    return table, series


def compute_overview(window_s: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Builds:
    - mesh status header (mesh-level)
    - node tiles (mesh-time centered NOW, offset/rate/freshness, converged state+reason)
    - offenders
    - link quality for topology (sigma med)
    """
    now = time.time()
    eff_window = max(window_s, CONV_WINDOW_S)

    node_rows = fetch_node_rows(eff_window, MAX_NODE_POINTS)
    link_rows = fetch_link_rows(eff_window, MAX_LINK_POINTS)

    # group node-only by node
    by_node: Dict[str, List[sqlite3.Row]] = {}
    latest_node: Dict[str, sqlite3.Row] = {}
    for r in node_rows:
        nid = str(row_get(r, "node_id", "") or "")
        if not nid:
            continue
        by_node.setdefault(nid, []).append(r)
        latest_node[nid] = r

    # compute latest t_mesh consensus centered (ms)
    tmesh_latest_ms: Dict[str, float] = {}
    # heuristic scaling like in timeseries
    tmesh_vals = []
    for r in latest_node.values():
        tm = _f(row_get(r, "t_mesh"))
        if tm is not None:
            tmesh_vals.append(abs(tm))
    scale = 1000.0
    med = _median(tmesh_vals) if tmesh_vals else None
    if med is not None and med > 1e11:
        scale = 1.0

    for nid, r in latest_node.items():
        tm = _f(row_get(r, "t_mesh"))
        if tm is not None:
            tmesh_latest_ms[nid] = tm * scale

    consensus_now = _median(list(tmesh_latest_ms.values())) if tmesh_latest_ms else 0.0

    # link sigma med in conv window per directed link
    conv_cut = now - CONV_WINDOW_S
    sig_by_link: Dict[str, List[float]] = {}
    last_link_seen: Dict[str, float] = {}
    for r in link_rows:
        t = _f(row_get(r, "created_at"))
        nid = str(row_get(r, "node_id", "") or "")
        pid = str(row_get(r, "peer_id", "") or "")
        if t is None or not nid or not pid:
            continue
        lid = f"{nid}→{pid}"
        last_link_seen[lid] = max(last_link_seen.get(lid, 0.0), t)
        if t >= conv_cut:
            sg = _f(row_get(r, "sigma_ms"))
            if sg is not None:
                sig_by_link.setdefault(lid, []).append(sg)

    sigma_med_link: Dict[str, Optional[float]] = {
        lid: _median(vals) for lid, vals in sig_by_link.items()
    }

    # thresholds
    fresh_thr = _fresh_thresh_s(cfg)

    # per-node convergence
    nodes_out: List[Dict[str, Any]] = []
    offenders = {
        "worst_node_delta": None,
        "stalest_node": None,
        "worst_link_sigma": None,
        "most_slew_clipped": None,
    }
    worst_delta = -1.0
    worst_age = -1.0
    worst_sigma = -1.0
    worst_clip = -1.0

    for nid in sorted(latest_node.keys()):
        r_last = latest_node[nid]
        t_last = _f(row_get(r_last, "created_at"))
        age_s = (now - t_last) if t_last is not None else None

        # mesh-time centered now
        centered_ms = None
        if nid in tmesh_latest_ms:
            centered_ms = tmesh_latest_ms[nid] - (consensus_now or 0.0)

        # controller stats in conv window
        recent = []
        for r in by_node.get(nid, []):
            t = _f(row_get(r, "created_at"))
            if t is not None and t >= conv_cut:
                recent.append(r)

        abs_applied = []
        clipped = []
        for rr in recent:
            da = _f(row_get(rr, "delta_applied_ms"))
            if da is not None:
                abs_applied.append(abs(da))
            sc = row_get(rr, "slew_clipped")
            try:
                if sc is not None:
                    clipped.append(1 if int(sc) else 0)
            except Exception:
                pass

        med_abs_applied = _median(abs_applied)
        clip_rate = (sum(clipped) / len(clipped)) if clipped else None

        # offset rate estimate (ms/s): from centered mesh-time last ~2 samples, robust-ish
        # Use t_mesh centered NOW and previous bin approx via last 2 samples available in recent
        rate_ms_s = None
        if len(recent) >= 2:
            r1 = recent[-1]
            r0 = recent[-2]
            t1 = _f(row_get(r1, "created_at"))
            t0 = _f(row_get(r0, "created_at"))
            tm1 = _f(row_get(r1, "t_mesh"))
            tm0 = _f(row_get(r0, "t_mesh"))
            if t1 is not None and t0 is not None and tm1 is not None and tm0 is not None and t1 > t0:
                rate_ms_s = ((tm1 - tm0) * scale * 1.0) / (t1 - t0)

        # link sigma med for node: collect outgoing link medians
        out_sigs = []
        for lid, sg_med in sigma_med_link.items():
            if lid.startswith(nid + "→") and sg_med is not None:
                out_sigs.append(float(sg_med))
        node_link_sigma_med = _median(out_sigs)

        # rules
        enough = len(recent) >= MIN_SAMPLES_WARMUP
        fresh_ok = (age_s is not None and age_s <= fresh_thr)
        delta_ok = (med_abs_applied is not None and med_abs_applied <= T_DELTA_APPLIED_MED_MS)
        link_ok = (node_link_sigma_med is None) or (node_link_sigma_med <= T_LINK_SIGMA_MED_MS)
        clip_ok = (clip_rate is None) or (clip_rate <= T_SLEW_CLIP_RATE)

        if not enough:
            state = "YELLOW"
            reason = "warming up"
        elif not fresh_ok:
            state = "RED"
            reason = f"stale data (age {age_s:.1f}s)"
        elif delta_ok and link_ok and clip_ok:
            state = "GREEN"
            reason = "converged"
        else:
            state = "YELLOW"
            rs = []
            if not delta_ok:
                rs.append(f"|Δapplied|med {med_abs_applied:.2f}ms" if med_abs_applied is not None else "|Δapplied|med n/a")
            if not link_ok:
                rs.append(f"linkσmed {node_link_sigma_med:.2f}ms")
            if not clip_ok and clip_rate is not None:
                rs.append(f"slew_clipped {clip_rate*100:.0f}%")
            reason = ", ".join(rs) if rs else "not converged"

        # offenders tracking
        if med_abs_applied is not None and med_abs_applied > worst_delta:
            worst_delta = med_abs_applied
            offenders["worst_node_delta"] = nid
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
            "age_s": age_s,
            "mesh_centered_ms": centered_ms,
            "mesh_consensus_ms": consensus_now,
            "med_abs_delta_applied_ms": med_abs_applied,
            "slew_clip_rate": clip_rate,
            "link_sigma_med_ms": node_link_sigma_med,
            "mesh_rate_ms_s": rate_ms_s,
            "last_seen_utc": _utc(t_last),
        })

    # worst link sigma (use medians if possible; otherwise last seen sigma in conv window)
    for lid, sg_med in sigma_med_link.items():
        if sg_med is not None and sg_med > worst_sigma:
            worst_sigma = sg_med
            offenders["worst_link_sigma"] = lid

    # mesh-level state: all green => green; any red with stale => red; else yellow
    mesh_state = "GREEN" if nodes_out and all(n["state"] == "GREEN" for n in nodes_out) else "YELLOW"
    if any(n["state"] == "RED" for n in nodes_out):
        mesh_state = "RED"

    # mesh-level reason: summarize worst offenders deterministically
    mesh_reason = "converged" if mesh_state == "GREEN" else ""
    if mesh_state != "GREEN":
        pieces = []
        if offenders["stalest_node"]:
            pieces.append(f"stale {offenders['stalest_node']}")
        if offenders["worst_node_delta"]:
            pieces.append(f"correction {offenders['worst_node_delta']}")
        if offenders["worst_link_sigma"]:
            pieces.append(f"linkσ {offenders['worst_link_sigma']}")
        mesh_reason = ", ".join(pieces) if pieces else "not converged"

    return {
        "mesh": {
            "state": mesh_state,
            "reason": mesh_reason,
            "now_utc": _utc(now),
            "conv_window_s": CONV_WINDOW_S,
            "thresholds": {
                "fresh_s": fresh_thr,
                "delta_applied_med_ms": T_DELTA_APPLIED_MED_MS,
                "link_sigma_med_ms": T_LINK_SIGMA_MED_MS,
                "slew_clip_rate": T_SLEW_CLIP_RATE,
                "min_samples_warmup": MIN_SAMPLES_WARMUP,
            },
        },
        "nodes": nodes_out,
        "offenders": offenders,
        "link_sigma_med": sigma_med_link,   # for topology coloring
        "link_last_seen": last_link_seen,
    }


def build_heatmap(metric: str, link_series: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Returns heatmap data suitable for matrix chart:
      [{link, t, v}]
    metric in {theta_ms, sigma_ms, rtt_ms, freshness_s}
    freshness uses per-link point age derived from series timestamps (relative to now)
    """
    now = time.time()
    data = []
    for lid, pts in link_series.items():
        for p in pts:
            t = _f(p.get("t"))
            if t is None:
                continue
            if metric == "freshness_s":
                v = now - t
            else:
                v = p.get(metric)
            if v is None:
                continue
            try:
                vv = float(v)
            except Exception:
                continue
            data.append({"link": lid, "t": float(t), "v": vv})

    # determine approximate bin count (unique time buckets)
    times = sorted({d["t"] for d in data})
    return {"data": data, "n_bins": len(times)}


# ------------------------------------------------------------
# Routes / API
# ------------------------------------------------------------
@app.route("/")
def index():
    topo = get_topology()
    return render_template_string(TEMPLATE, topo=topo, db_path=str(DB_PATH))


@app.route("/api/overview")
def api_overview():
    try:
        cfg = load_config()
        window_s = float((cfg.get("sync", {}) or {}).get("ui_window_s", WINDOW_S_DEFAULT))
        return jsonify(compute_overview(window_s, cfg))
    except Exception as e:
        return _json_error(f"/api/overview failed: {e}")


@app.route("/api/mesh_timeseries")
def api_mesh_timeseries():
    try:
        cfg = load_config()
        sync = cfg.get("sync", {}) or {}
        window_s = float(sync.get("ui_window_s", WINDOW_S_DEFAULT))
        bin_s = float(sync.get("ui_bin_s", BIN_S_DEFAULT))
        series = build_node_timeseries_mesh(window_s, bin_s)
        return jsonify({"series": series, "meta": {"window_s": window_s, "bin_s": bin_s, "x_axis": "created_at"}})
    except Exception as e:
        return _json_error(f"/api/mesh_timeseries failed: {e}")


@app.route("/api/links")
def api_links():
    try:
        cfg = load_config()
        sync = cfg.get("sync", {}) or {}
        window_s = float(sync.get("ui_window_s", WINDOW_S_DEFAULT))
        bin_s = float(sync.get("ui_bin_s", BIN_S_DEFAULT))
        table, series = build_links_table_and_series(window_s, bin_s)
        return jsonify({"table": table, "series": series, "meta": {"window_s": window_s, "bin_s": bin_s}})
    except Exception as e:
        return _json_error(f"/api/links failed: {e}")


@app.route("/api/heatmap")
def api_heatmap():
    try:
        metric = str(request.args.get("metric", "sigma_ms"))
        if metric not in {"theta_ms", "sigma_ms", "rtt_ms", "freshness_s"}:
            metric = "sigma_ms"
        cfg = load_config()
        sync = cfg.get("sync", {}) or {}
        window_s = float(sync.get("ui_window_s", WINDOW_S_DEFAULT))
        bin_s = float(sync.get("ui_bin_s", BIN_S_DEFAULT))
        _table, series = build_links_table_and_series(window_s, bin_s)
        hm = build_heatmap(metric, series)
        return jsonify({"metric": metric, "heatmap": hm, "meta": {"window_s": window_s, "bin_s": bin_s}})
    except Exception as e:
        return _json_error(f"/api/heatmap failed: {e}")


@app.route("/api/controller")
def api_controller():
    try:
        cfg = load_config()
        sync = cfg.get("sync", {}) or {}
        window_s = float(sync.get("ui_window_s", WINDOW_S_DEFAULT))
        ctrl = build_controller_timeseries(window_s)
        return jsonify({"controller": ctrl, "meta": {"window_s": window_s}})
    except Exception as e:
        return _json_error(f"/api/controller failed: {e}")


# ------------------------------------------------------------
# UI Template (one page)
# ------------------------------------------------------------
TEMPLATE = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>MeshTime Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <style>
    :root{
      --bg:#0f0f10; --card:#171718; --muted:rgba(255,255,255,0.65);
      --grid:rgba(255,255,255,0.08);
      --ok:#2ecc71; --warn:#f1c40f; --bad:#e74c3c;
    }
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin:0; background:var(--bg); color:#eee; padding: 1.25rem; }
    h1,h2,h3 { margin:0; }
    .sub{ margin-top:0.35rem; color:var(--muted); font-size:0.9rem; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .row { display:flex; gap:0.75rem; align-items:center; flex-wrap:wrap; }
    .sp { justify-content:space-between; }
    .card { background:var(--card); border-radius:14px; padding:1rem 1.15rem; box-shadow: 0 0 0 1px rgba(255,255,255,0.05); }
    .grid { display:grid; grid-template-columns: minmax(0,1.2fr) minmax(0,2fr); gap: 1rem; margin-top: 1rem; }
    @media (max-width: 1200px) { .grid { grid-template-columns: minmax(0,1fr); } }
    .pill { display:inline-block; padding:0.12rem 0.55rem; border-radius:999px; font-size:0.75rem; font-weight:700; }
    .ok { background: rgba(46,204,113,0.14); color: var(--ok); }
    .warn { background: rgba(241,196,15,0.14); color: var(--warn); }
    .bad { background: rgba(231,76,60,0.14); color: var(--bad); }

    .kpiGrid { display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:0.75rem; margin-top: 0.75rem; }
    @media (max-width: 1100px) { .kpiGrid { grid-template-columns: repeat(2, minmax(0,1fr)); } }
    @media (max-width: 700px) { .kpiGrid { grid-template-columns: minmax(0,1fr); } }
    .kpi { background: rgba(255,255,255,0.03); border-radius: 12px; padding:0.75rem; box-shadow: 0 0 0 1px rgba(255,255,255,0.03); }
    .kpi .big { font-size: 1.25rem; font-weight: 800; margin-top:0.15rem; }
    .kpi .small { margin-top:0.2rem; color: var(--muted); font-size: 0.85rem; }
    .kpi .reason { margin-top:0.35rem; color:rgba(255,255,255,0.72); font-size:0.85rem; }

    canvas { max-width:100%; }
    .plots { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin-top:0.75rem; }
    @media (max-width: 1200px) { .plots { grid-template-columns: minmax(0,1fr); } }

    table { width:100%; border-collapse: collapse; font-size:0.92rem; margin-top:0.6rem; }
    th, td { padding: 0.35rem 0.5rem; text-align:left; vertical-align:top; }
    th { border-bottom: 1px solid rgba(255,255,255,0.14); font-weight:750; }
    tr:nth-child(even) td { background: rgba(255,255,255,0.02); }

    #meshTopo { width:100%; height: 220px; background:#131313; border-radius: 12px; margin-top:0.6rem; }

    .toggle { display:flex; gap:0.6rem; align-items:center; }
    .toggle input { transform: scale(1.2); }

    .debugOnly { display:none; }
    body.debug .debugOnly { display:block; }

    .controls { display:flex; gap:0.75rem; align-items:center; flex-wrap:wrap; }
    select, button {
      background:#111; color:#eee; border: 1px solid rgba(255,255,255,0.15);
      border-radius: 10px; padding: 0.35rem 0.55rem;
    }
    .muted{ color: var(--muted); }
  </style>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@1.3.0/dist/chartjs-chart-matrix.min.js"></script>
</head>

<body>
  <div class="row sp">
    <div>
      <h1>MeshTime</h1>
      <div class="sub">
        DB: <span class="mono">{{ db_path }}</span> · X-Achse: <b>created_at (Sink)</b>
      </div>
    </div>

    <div class="row">
      <div class="toggle card" style="padding:0.55rem 0.75rem;">
        <input id="debugToggle" type="checkbox">
        <label for="debugToggle" class="muted">Debug</label>
      </div>
    </div>
  </div>

  <!-- 10-second header -->
  <div class="card" style="margin-top:1rem;">
    <div class="row sp">
      <div class="row">
        <span id="meshStatePill" class="pill warn">…</span>
        <div>
          <div style="font-weight:800; font-size:1.05rem;" id="meshHeaderLine">lade…</div>
          <div class="sub" id="meshHeaderMeta">lade…</div>
        </div>
      </div>
      <div class="controls">
        <span class="muted">Heatmap:</span>
        <select id="heatMetric">
          <option value="sigma_ms">σ (stability)</option>
          <option value="theta_ms">θ (offset)</option>
          <option value="rtt_ms">RTT</option>
          <option value="freshness_s">freshness</option>
        </select>
      </div>
    </div>

    <div class="kpiGrid" id="nodeTiles">
      <div class="kpi"><div class="muted">Nodes</div><div class="big">…</div></div>
    </div>
  </div>

  <div class="grid">
    <!-- Left column: topology + offenders -->
    <div>
      <div class="card">
        <h2 style="font-size:1.05rem;">Topologie</h2>
        <canvas id="meshTopo"></canvas>
        <div class="sub">Node-Farbe = Node-Ampel. Link-Dicke/Farbe = σ (Median im Conv-Window, wenn vorhanden).</div>
      </div>

      <div class="card" style="margin-top:1rem;">
        <h2 style="font-size:1.05rem;">Worst Offenders</h2>
        <div class="sub" id="offendersLine">lade…</div>
      </div>

      <div class="card debugOnly" style="margin-top:1rem;">
        <h2 style="font-size:1.05rem;">Thresholds</h2>
        <div class="sub mono" id="thrLine">lade…</div>
      </div>
    </div>

    <!-- Right column: plots + tables -->
    <div>
      <div class="card">
        <div class="row sp">
          <h2 style="font-size:1.05rem;">t_mesh(node) − Konsens über Zeit</h2>
          <div class="sub">Centered (Median-Konsens pro Bin). Einheit: ms.</div>
        </div>
        <canvas id="meshChart" height="190"></canvas>
      </div>

      <div class="card" style="margin-top:1rem;">
        <div class="row sp">
          <h2 style="font-size:1.05rem;">Link Metrics (Conv-Window)</h2>
          <div class="sub muted">Directed Link-ID: A→B</div>
        </div>
        <table id="linksTable">
          <thead>
            <tr>
              <th>Link</th>
              <th>State</th>
              <th>Age</th>
              <th>θ med</th>
              <th>RTT med</th>
              <th>σ med</th>
              <th>nσ</th>
            </tr>
          </thead>
          <tbody><tr><td colspan="7" class="muted">lade…</td></tr></tbody>
        </table>
      </div>

      <div class="card" style="margin-top:1rem;">
        <div class="row sp">
          <h2 style="font-size:1.05rem;">Heatmap</h2>
          <div class="sub">Genau 1 Metrik (Dropdown oben). Fixes Layout, keine abgeschnittenen Labels.</div>
        </div>
        <canvas id="heatmap" height="190"></canvas>
      </div>

      <div class="card debugOnly" style="margin-top:1rem;">
        <div class="row sp">
          <div>
            <h2 style="font-size:1.05rem;">Controller (Debug)</h2>
            <div class="sub">delta_desired/applied, dt, slew_clipped → Ursache/Wirkung</div>
          </div>
          <div class="controls">
            <span class="muted">Node:</span>
            <select id="ctrlNode"></select>
          </div>
        </div>

        <div class="plots">
          <div><div class="sub">delta_desired_ms</div><canvas id="c_des" height="140"></canvas></div>
          <div><div class="sub">delta_applied_ms</div><canvas id="c_app" height="140"></canvas></div>
          <div><div class="sub">dt_s</div><canvas id="c_dt" height="140"></canvas></div>
          <div><div class="sub">slew_clipped (0/1)</div><canvas id="c_clip" height="140"></canvas></div>
        </div>
      </div>

    </div>
  </div>

<script>
  const COLORS = ['#2ecc71','#3498db','#f1c40f','#e74c3c','#9b59b6','#1abc9c','#e67e22'];

  function pillClass(state){
    if(state==='GREEN') return 'pill ok';
    if(state==='RED') return 'pill bad';
    return 'pill warn';
  }

  function setMeshHeader(state, line, meta){
    const pill = document.getElementById('meshStatePill');
    pill.className = pillClass(state);
    pill.textContent = state;
    document.getElementById('meshHeaderLine').textContent = line || '';
    document.getElementById('meshHeaderMeta').textContent = meta || '';
  }

  async function fetchJson(url){
    const r = await fetch(url);
    const j = await r.json().catch(()=>({error:'invalid json'}));
    if(!r.ok) throw new Error(j.error || 'http error');
    return j;
  }

  // Charts
  let meshChart, heatChart, cDes, cApp, cDt, cClip;

  function mkLine(ctx, yLabel){
    return new Chart(ctx, {
      type:'line',
      data:{datasets:[]},
      options:{
        responsive:true,
        animation:false,
        scales:{
          x:{type:'time', ticks:{color:'#aaa'}, grid:{color:'rgba(255,255,255,0.06)'}},
          y:{ticks:{color:'#aaa'}, grid:{color:'rgba(255,255,255,0.06)'}, title:{display:true, text:yLabel, color:'#aaa'}}
        },
        plugins:{ legend:{labels:{color:'#eee'}} },
        elements:{point:{radius:0}, line:{tension:0.05}}
      }
    });
  }

  function mkHeatmap(ctx){
    return new Chart(ctx, {
      type: 'matrix',
      data: { datasets: [{
        label: 'Heatmap',
        data: [],
        borderWidth: 0,
        backgroundColor: (context) => {
          const raw = context.raw;
          if(!raw || typeof raw.v !== 'number') return 'rgba(0,0,0,0)';
          const maxV = context.chart._maxV || 1;
          const ratio = Math.min(1, raw.v / maxV);
          const r = Math.round(255 * ratio);
          const g = Math.round(255 * (1 - ratio));
          return `rgba(${r},${g},160,0.85)`;
        },
        width: (context) => {
          const area = context.chart.chartArea || {};
          const n = context.chart._nBins || 1;
          const w = (area.right - area.left) / n;
          return (Number.isFinite(w) && w > 0) ? w : 10;
        },
        height: (context) => {
          const area = context.chart.chartArea || {};
          const cats = context.chart._cats || ['x'];
          const h = (area.bottom - area.top) / (cats.length || 1);
          return (Number.isFinite(h) && h > 0) ? h : 10;
        }
      }]},
      options: {
        responsive:true,
        animation:false,
        scales: {
          x: { type:'time', ticks:{color:'#aaa'}, grid:{color:'rgba(255,255,255,0.06)'} },
          y: {
            type:'linear',
            ticks:{
              color:'#aaa',
              callback: function(v){
                const cats = this.chart._cats || [];
                return cats[Math.round(v)] || '';
              }
            },
            grid:{color:'rgba(255,255,255,0.06)'}
          }
        },
        plugins:{
          legend:{labels:{color:'#eee'}},
          tooltip:{callbacks:{label:(ctx)=>{
            const cats = ctx.chart._cats || [];
            const lid = cats[Math.round(ctx.raw.y)] || '?';
            const t = new Date(ctx.raw.x);
            return `${lid} @ ${t.toLocaleTimeString()} : ${ctx.raw.v.toFixed(2)}`;
          }}}
        }
      }
    });
  }

  function updateMeshChart(series){
    const ids = Object.keys(series||{}).sort();
    meshChart.data.datasets = [];
    ids.forEach((id, i)=>{
      const c = COLORS[i % COLORS.length];
      const pts = (series[id]||[]).map(p=>({x:new Date(p.t*1000), y:p.mesh_ms}));
      meshChart.data.datasets.push({label:`Node ${id}`, data:pts, borderColor:c, borderWidth:1.6});
    });
    meshChart.update();
  }

  function renderLinksTable(table){
    const tb = document.querySelector('#linksTable tbody');
    tb.innerHTML = '';
    if(!table || !table.length){
      tb.innerHTML = `<tr><td colspan="7" class="muted">keine Link-Daten</td></tr>`;
      return;
    }
    table.forEach(r=>{
      const st = r.state || 'DOWN';
      const pill = st==='UP' ? 'pill ok' : 'pill bad';
      const age = (r.age_s==null) ? 'n/a' : `${r.age_s.toFixed(1)} s`;
      const th = (r.theta_med_ms==null) ? '—' : `${r.theta_med_ms.toFixed(2)} ms`;
      const rt = (r.rtt_med_ms==null) ? '—' : `${r.rtt_med_ms.toFixed(2)} ms`;
      const sg = (r.sigma_med_ms==null) ? '—' : `${r.sigma_med_ms.toFixed(2)} ms`;
      const ns = (r.n_sigma==null) ? '—' : r.n_sigma;

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="mono">${r.link}</td>
        <td><span class="${pill}">${st}</span></td>
        <td>${age}</td>
        <td>${th}</td>
        <td>${rt}</td>
        <td>${sg}</td>
        <td>${ns}</td>
      `;
      tb.appendChild(tr);
    });
  }

  function updateHeatmap(metric, hm){
    const data = (hm && hm.data) ? hm.data : [];
    if(!data.length){
      heatChart.data.datasets[0].data = [];
      heatChart._cats = [];
      heatChart._nBins = 1;
      heatChart._maxV = 1;
      heatChart.update();
      return;
    }
    const cats = Array.from(new Set(data.map(d=>d.link))).sort();
    const idx = new Map(cats.map((c,i)=>[c,i]));
    let maxV = 0;
    const matrix = data.map(d=>{
      const v = Math.abs(d.v);
      if(v > maxV) maxV = v;
      return {x:new Date(d.t*1000), y:(idx.get(d.link) ?? 0), v:v};
    });
    heatChart.data.datasets[0].data = matrix;
    heatChart._cats = cats;
    heatChart._nBins = hm.n_bins || 1;
    heatChart._maxV = maxV || 1;
    heatChart.update();
  }

  function updateController(nodeId, ctrl){
    function one(chart, key){
      const pts = (ctrl[nodeId]||[]).filter(p=>p[key]!=null).map(p=>({x:new Date(p.t*1000), y:p[key]}));
      chart.data.datasets = [{label:`Node ${nodeId}`, data:pts, borderColor:'#2ecc71', borderWidth:1.6}];
      chart.update();
    }
    one(cDes,'delta_desired_ms');
    one(cApp,'delta_applied_ms');
    one(cDt,'dt_s');
    one(cClip,'slew_clipped');
  }

  function drawTopology(canvas, topo, nodeStates, linkSigmaMed){
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);

    const nodes = topo.nodes || [];
    const links = topo.links || [];
    if(!nodes.length){
      ctx.fillStyle = '#777';
      ctx.font = '12px system-ui';
      ctx.fillText('Keine Nodes in config/nodes.json', 10, 20);
      return;
    }

    const n = nodes.length;
    const R = Math.min(w,h) * 0.35;
    const cx = w/2, cy = h/2;

    const pos = {};
    nodes.forEach((nd,i)=>{
      const a = (2*Math.PI*i)/n - Math.PI/2;
      pos[nd.id] = {x: cx + R*Math.cos(a), y: cy + R*Math.sin(a)};
    });

    // link drawing: use sigma median if known (directed links)
    function linkKey(a,b){ return `${a}→${b}`; }

    links.forEach(L=>{
      const a = pos[L.a], b = pos[L.b];
      if(!a || !b) return;

      // sigma: take max of both directions if available
      const s1 = linkSigmaMed[linkKey(L.a, L.b)];
      const s2 = linkSigmaMed[linkKey(L.b, L.a)];
      const sigma = (s1!=null && s2!=null) ? Math.max(s1,s2) : (s1!=null? s1 : (s2!=null? s2 : null));

      let stroke = 'rgba(255,255,255,0.18)';
      let lw = 1.0;
      if(sigma!=null){
        if(sigma <= 2.0){ stroke = 'rgba(46,204,113,0.55)'; lw = 2.4; }
        else if(sigma <= 5.0){ stroke = 'rgba(241,196,15,0.55)'; lw = 2.0; }
        else { stroke = 'rgba(231,76,60,0.55)'; lw = 2.0; }
      }

      ctx.strokeStyle = stroke;
      ctx.lineWidth = lw;
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
    });

    // nodes
    nodes.forEach(nd=>{
      const p = pos[nd.id]; if(!p) return;
      const st = nodeStates[nd.id] || 'YELLOW';
      const isRoot = !!nd.is_root;
      const r = isRoot ? 20 : 16;

      let fill='rgba(241,196,15,0.14)', stroke='#f1c40f';
      if(st==='GREEN'){ fill='rgba(46,204,113,0.18)'; stroke='#2ecc71'; }
      if(st==='RED'){ fill='rgba(231,76,60,0.18)'; stroke='#e74c3c'; }

      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,2*Math.PI);
      ctx.fillStyle = fill; ctx.fill();
      ctx.lineWidth = isRoot ? 3.0 : 2.0;
      ctx.strokeStyle = stroke; ctx.stroke();

      ctx.fillStyle = '#ecf0f1';
      ctx.font = '12px system-ui';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(nd.id, p.x, p.y);
    });
  }

  function renderNodeTiles(nodes){
    const wrap = document.getElementById('nodeTiles');
    wrap.innerHTML = '';
    (nodes||[]).forEach((n, i)=>{
      const st = n.state || 'YELLOW';
      const cls = pillClass(st);

      const m = (n.mesh_centered_ms==null) ? 'n/a' : `${n.mesh_centered_ms.toFixed(2)} ms`;
      const age = (n.age_s==null) ? 'n/a' : `${n.age_s.toFixed(1)} s`;
      const da = (n.med_abs_delta_applied_ms==null) ? 'n/a' : `${n.med_abs_delta_applied_ms.toFixed(2)} ms`;
      const sig = (n.link_sigma_med_ms==null) ? '—' : `${n.link_sigma_med_ms.toFixed(2)} ms`;
      const clip = (n.slew_clip_rate==null) ? '—' : `${(n.slew_clip_rate*100).toFixed(0)}%`;
      const rate = (n.mesh_rate_ms_s==null) ? '—' : `${n.mesh_rate_ms_s.toFixed(3)} ms/s`;

      const div = document.createElement('div');
      div.className = 'kpi';
      div.innerHTML = `
        <div class="row sp">
          <div class="muted">Node ${n.node_id}</div>
          <span class="${cls}">${st}</span>
        </div>
        <div class="big">${m}</div>
        <div class="small">rate ${rate} · age ${age}</div>
        <div class="small">|Δapplied|med ${da} · clip ${clip} · linkσmed ${sig}</div>
        <div class="reason">${n.reason || ''}</div>
      `;
      wrap.appendChild(div);
    });
  }

  function applyDebug(){
    document.body.classList.toggle('debug', document.getElementById('debugToggle').checked);
  }

  async function refreshAll(){
    const topo = {{ topo|tojson }};

    const [ov, mesh, links, ctrl, hm] = await Promise.all([
      fetchJson('/api/overview'),
      fetchJson('/api/mesh_timeseries'),
      fetchJson('/api/links'),
      fetchJson('/api/controller'),
      fetchJson('/api/heatmap?metric=' + encodeURIComponent(document.getElementById('heatMetric').value)),
    ]);

    // header
    const M = ov.mesh || {};
    setMeshHeader(M.state || 'YELLOW', `${M.state||'…'}: ${M.reason||''}`, `now=${M.now_utc||'n/a'} · conv_window=${M.conv_window_s||'?'}` );

    // debug thresholds line
    const T = (M.thresholds||{});
    document.getElementById('thrLine').textContent =
      `fresh≤${(T.fresh_s??'?')}s  |Δapplied|med≤${(T.delta_applied_med_ms??'?')}ms  linkσmed≤${(T.link_sigma_med_ms??'?')}ms  clip≤${Math.round((T.slew_clip_rate??0)*100)}%  warmup≥${(T.min_samples_warmup??'?')} samples`;

    // node tiles
    renderNodeTiles(ov.nodes || []);

    // offenders
    const off = ov.offenders || {};
    document.getElementById('offendersLine').textContent =
      `worst |Δapplied|: ${off.worst_node_delta || '—'} · stalest: ${off.stalest_node || '—'} · most clipped: ${off.most_slew_clipped || '—'} · worst link σ: ${off.worst_link_sigma || '—'}`;

    // topology canvas sizing
    const c = document.getElementById('meshTopo');
    c.width = c.clientWidth;
    c.height = c.clientHeight;

    const nodeStates = {};
    (ov.nodes||[]).forEach(n => nodeStates[n.node_id] = n.state);
    drawTopology(c, topo, nodeStates, ov.link_sigma_med || {});

    // mesh plot
    updateMeshChart(mesh.series || {});

    // links table
    renderLinksTable(links.table || []);

    // heatmap
    updateHeatmap(hm.metric, hm.heatmap || {data:[], n_bins:0});

    // controller debug node selector
    const ctrlMap = ctrl.controller || {};
    const sel = document.getElementById('ctrlNode');
    const nodes = Object.keys(ctrlMap).sort();
    if(sel.options.length === 0){
      nodes.forEach(n=>{
        const o = document.createElement('option');
        o.value = n; o.textContent = n;
        sel.appendChild(o);
      });
    }
    const chosen = sel.value || (nodes[0] || '');
    if(chosen && ctrlMap[chosen]){
      updateController(chosen, ctrlMap);
    }
  }

  window.addEventListener('load', ()=>{
    // charts init
    meshChart = mkLine(document.getElementById('meshChart').getContext('2d'), 'ms');
    heatChart = mkHeatmap(document.getElementById('heatmap').getContext('2d'));

    cDes = mkLine(document.getElementById('c_des').getContext('2d'), 'ms');
    cApp = mkLine(document.getElementById('c_app').getContext('2d'), 'ms');
    cDt = mkLine(document.getElementById('c_dt').getContext('2d'), 's');
    cClip = mkLine(document.getElementById('c_clip').getContext('2d'), '0/1');

    document.getElementById('debugToggle').addEventListener('change', applyDebug);
    document.getElementById('heatMetric').addEventListener('change', ()=>refreshAll().catch(console.error));
    document.getElementById('ctrlNode').addEventListener('change', ()=>refreshAll().catch(console.error));

    applyDebug();
    refreshAll().catch(console.error);
    setInterval(()=>refreshAll().catch(console.error), 2000);
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
    print("MeshTime Dashboard on http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)
