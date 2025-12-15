# -*- coding: utf-8 -*-
# mesh/web_ui.py
# MeshTime Web Dashboard (FINAL)
#
# Key principle:
#   Use created_at (sink clock) as global axis.
#   Never use sender t_wall as global axis in unrooted mesh.
#
# DB:
#   ntp_reference:
#     - node-only samples: peer_id IS NULL
#     - link samples:     peer_id NOT NULL (theta_ms, rtt_ms, sigma_ms)
#
# HARD RULE:
#   This server must NEVER crash due to formatting None values.

from __future__ import annotations

import json
import math
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

from flask import Flask, render_template_string, jsonify, make_response

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "mesh_data.sqlite"
CFG_PATH = BASE_DIR / "config" / "nodes.json"

from mesh.storage import Storage  # ensures schema

app = Flask(__name__)

# -----------------------------
# UI / Aggregation Tuning
# -----------------------------
WINDOW_SECONDS_DEFAULT = 600        # 10 min
MAX_POINTS_DEFAULT = 8000           # ntp_reference read limit (node-only)
BIN_S_DEFAULT = 0.5                 # 500ms bins
HEATMAP_MAX_BINS = 40
JITTER_MIN_SAMPLES = 10

LINK_MAX_POINTS_DEFAULT = 12000
LINK_MIN_SAMPLES = 8

# Convergence thresholds (Operator mode)
CONV_WINDOW_S = 30.0
THRESH_FRESH_S = 3.0
THRESH_DELTA_APPLIED_MED_MS = 0.5
THRESH_SLEW_CLIP_RATE = 0.20
THRESH_LINK_SIGMA_MED_MS = 2.0


# ------------------------------------------------------------
# Helpers (must never crash)
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

    # cfg structure: { "A":{...}, "B":{...}, "sync":{...} }
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


def _utc(ts: Optional[float]) -> str:
    if ts is None:
        return "n/a"
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


def row_has(r: Any, key: str) -> bool:
    # sqlite3.Row supports keys(), dict supports "in"
    try:
        ks = r.keys()  # type: ignore[attr-defined]
        return key in ks
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


def fmt(x: Any, spec: str = ".2f", default: str = "n/a") -> str:
    v = _f(x)
    if v is None:
        return default
    try:
        return format(v, spec)
    except Exception:
        return default


def fmt_ms(x: Any, default: str = "n/a") -> str:
    s = fmt(x, ".2f", default)
    return s if s == default else f"{s} ms"


def fmt_s(x: Any, default: str = "n/a") -> str:
    s = fmt(x, ".1f", default)
    return s if s == default else f"{s} s"


def fmt_pct(x: Any, default: str = "n/a") -> str:
    s = fmt(x, ".0f", default)
    return s if s == default else f"{s}%"


def clamp(v: float, lo: float, hi: float) -> float:
    try:
        return max(float(lo), min(float(hi), float(v)))
    except Exception:
        return float(lo)


# ------------------------------------------------------------
# Robust stats (IQR/MAD + quantiles)
# ------------------------------------------------------------
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


def robust_iqr(values: List[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    q1 = _quantile_sorted(xs, 0.25)
    q3 = _quantile_sorted(xs, 0.75)
    return float(q3 - q1)


def robust_mad(values: List[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    med = _quantile_sorted(xs, 0.50)
    abs_dev = sorted([abs(v - med) for v in values])
    mad = _quantile_sorted(abs_dev, 0.50)
    return float(mad)


def robust_median(values: List[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    return float(_quantile_sorted(xs, 0.50))


# ------------------------------------------------------------
# DB Readers (node-only vs link rows) - robust to missing columns
# ------------------------------------------------------------
def _select_existing(cols: Set[str], wanted: List[str]) -> List[str]:
    return [c for c in wanted if c in cols]


def fetch_node_rows(window_s: float, limit: int) -> List[sqlite3.Row]:
    conn = get_conn()
    try:
        cols = _table_cols(conn, "ntp_reference")
        if "node_id" not in cols or "created_at" not in cols:
            return []

        wanted = [
            "id", "node_id", "created_at", "offset",
            "peer_id",
            "delta_desired_ms", "delta_applied_ms", "dt_s", "slew_clipped",
            "t_wall", "t_mesh", "t_mono", "err_mesh_vs_wall",
        ]
        sel = _select_existing(cols, wanted)
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
        needed = {"peer_id", "theta_ms", "rtt_ms", "sigma_ms", "created_at", "node_id"}
        if not needed.issubset(cols):
            return []

        cutoff = time.time() - float(window_s)
        cur = conn.cursor()
        return cur.execute(
            """
            SELECT id, node_id, peer_id, created_at, theta_ms, rtt_ms, sigma_ms
            FROM ntp_reference
            WHERE created_at >= ?
              AND peer_id IS NOT NULL
              AND (theta_ms IS NOT NULL OR rtt_ms IS NOT NULL OR sigma_ms IS NOT NULL)
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (cutoff, int(limit)),
        ).fetchall()
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ------------------------------------------------------------
# Overview + Convergence
# ------------------------------------------------------------
def compute_overview(window_s: float = WINDOW_SECONDS_DEFAULT) -> Dict[str, Any]:
    now = time.time()
    eff_window = max(float(window_s), float(CONV_WINDOW_S))

    node_rows = fetch_node_rows(window_s=eff_window, limit=MAX_POINTS_DEFAULT)
    link_rows = fetch_link_rows(window_s=eff_window, limit=LINK_MAX_POINTS_DEFAULT)

    latest: Dict[str, sqlite3.Row] = {}
    by_node_recent: Dict[str, List[sqlite3.Row]] = {}

    for r in node_rows:
        nid = str(row_get(r, "node_id", "") or "")
        if not nid:
            continue
        by_node_recent.setdefault(nid, []).append(r)
        latest[nid] = r

    # link sigmas per directed link
    link_sigmas: Dict[str, List[float]] = {}
    link_latest_sigma: Dict[str, float] = {}
    for r in link_rows:
        nid = str(row_get(r, "node_id", "") or "")
        pid = str(row_get(r, "peer_id", "") or "")
        if not nid or not pid:
            continue
        lid = f"{nid}->{pid}"
        sig = _f(row_get(r, "sigma_ms"))
        if sig is not None:
            link_sigmas.setdefault(lid, []).append(sig)
            link_latest_sigma[lid] = sig

    # mesh median of latest offsets
    latest_offsets_ms: List[float] = []
    for r in latest.values():
        off = _f(row_get(r, "offset"))
        if off is not None:
            latest_offsets_ms.append(off * 1000.0)
    mesh_median_ms = robust_median(latest_offsets_ms) if latest_offsets_ms else 0.0

    offenders: Dict[str, Any] = {
        "worst_node_delta": None,
        "worst_node_freshness": None,
        "worst_link_sigma": None,
        "most_slew_clipped": None,
    }

    worst_delta = -1.0
    worst_age = -1.0
    worst_sigma = -1.0
    worst_clip_rate = -1.0

    conv_cut = now - float(CONV_WINDOW_S)
    nodes_out: List[Dict[str, Any]] = []

    for nid, r in sorted(latest.items(), key=lambda kv: kv[0]):
        created_at = _f(row_get(r, "created_at"))
        age_s = (now - created_at) if created_at is not None else None

        off = _f(row_get(r, "offset"))
        offset_ms = (off * 1000.0) if off is not None else None
        centered_ms = (offset_ms - mesh_median_ms) if offset_ms is not None else None

        # controller stats in last CONV_WINDOW_S
        recent: List[sqlite3.Row] = []
        for x in by_node_recent.get(nid, []):
            tx = _f(row_get(x, "created_at"))
            if tx is not None and tx >= conv_cut:
                recent.append(x)

        deltas_applied: List[float] = []
        clipped: List[int] = []
        for x in recent:
            da = _f(row_get(x, "delta_applied_ms"))
            if da is not None:
                deltas_applied.append(abs(da))

            sc = row_get(x, "slew_clipped")
            try:
                if sc is not None:
                    clipped.append(1 if int(sc) else 0)
            except Exception:
                pass

        med_abs_delta_applied_ms = robust_median(deltas_applied) if deltas_applied else None
        clip_rate = (sum(clipped) / len(clipped)) if clipped else None

        # link sigma median from nid -> *
        sigs_from_node: List[float] = []
        for lid, sigs in link_sigmas.items():
            if lid.startswith(nid + "->") and sigs:
                sigs_from_node.extend(sigs)
        link_sigma_med = robust_median(sigs_from_node) if sigs_from_node else None

        # convergence rules
        fresh_ok = (age_s is not None and age_s <= THRESH_FRESH_S)
        delta_ok = (med_abs_delta_applied_ms is not None and med_abs_delta_applied_ms <= THRESH_DELTA_APPLIED_MED_MS)
        clip_ok = (clip_rate is None) or (clip_rate <= THRESH_SLEW_CLIP_RATE)
        link_ok = (link_sigma_med is None) or (link_sigma_med <= THRESH_LINK_SIGMA_MED_MS)
        enough_samples = len(recent) >= 3

        if not enough_samples:
            state = "YELLOW"
            reason = "warming up / too few samples"
        elif not fresh_ok:
            state = "RED"
            reason = f"stale data (age {fmt(age_s,'.1f','n/a')}s)"
        elif delta_ok and clip_ok and link_ok:
            state = "GREEN"
            reason = "converged"
        else:
            state = "YELLOW"
            reasons: List[str] = []
            if not delta_ok:
                reasons.append(f"delta_applied_med {fmt(med_abs_delta_applied_ms,'.2f','n/a')}ms")
            if not clip_ok and clip_rate is not None:
                reasons.append(f"slew_clipped {fmt(clip_rate * 100.0,'.0f','n/a')}%")
            if not link_ok and link_sigma_med is not None:
                reasons.append(f"link_sigma_med {fmt(link_sigma_med,'.2f','n/a')}ms")
            reason = ", ".join(reasons) if reasons else "not converged"

        # offenders
        if med_abs_delta_applied_ms is not None and med_abs_delta_applied_ms > worst_delta:
            worst_delta = med_abs_delta_applied_ms
            offenders["worst_node_delta"] = nid
        if age_s is not None and age_s > worst_age:
            worst_age = age_s
            offenders["worst_node_freshness"] = nid
        if clip_rate is not None and clip_rate > worst_clip_rate:
            worst_clip_rate = clip_rate
            offenders["most_slew_clipped"] = nid

        nodes_out.append({
            "node_id": nid,
            "last_seen_created_at": created_at,
            "last_seen_utc": _utc(created_at),
            "age_s": age_s,
            "offset_ms": offset_ms,
            "offset_centered_ms": centered_ms,
            "mesh_median_ms": mesh_median_ms,
            "med_abs_delta_applied_ms": med_abs_delta_applied_ms,
            "slew_clip_rate": clip_rate,
            "link_sigma_med_ms": link_sigma_med,
            "state": state,
            "reason": reason,
        })

    # worst link sigma
    for lid, sig in link_latest_sigma.items():
        if sig is not None and sig > worst_sigma:
            worst_sigma = sig
            offenders["worst_link_sigma"] = lid

    return {
        "nodes": nodes_out,
        "offenders": offenders,
        "meta": {
            "now": now,
            "now_utc": _utc(now),
            "conv_window_s": CONV_WINDOW_S,
            "thresholds": {
                "fresh_s": THRESH_FRESH_S,
                "delta_applied_med_ms": THRESH_DELTA_APPLIED_MED_MS,
                "slew_clip_rate": THRESH_SLEW_CLIP_RATE,
                "link_sigma_med_ms": THRESH_LINK_SIGMA_MED_MS,
            },
        },
    }


# ------------------------------------------------------------
# Status snapshot (node-only; Δ vs ref from offsets)
# ------------------------------------------------------------
def get_status_snapshot(reference_node: str = "C") -> Dict[str, Any]:
    conn = get_conn()
    try:
        cols = _table_cols(conn, "ntp_reference")
        if not {"node_id", "created_at", "offset", "id"}.issubset(cols):
            return {"rows": [], "reference_node": reference_node}

        where_node_only = "WHERE peer_id IS NULL" if "peer_id" in cols else ""
        cur = conn.cursor()

        # pick latest row per node (node-only if peer_id exists)
        q = f"""
            SELECT r.*
            FROM ntp_reference r
            JOIN (
              SELECT node_id, MAX(id) AS max_id
              FROM ntp_reference
              {where_node_only}
              GROUP BY node_id
            ) t ON t.node_id = r.node_id AND t.max_id = r.id
            ORDER BY r.node_id
        """
        last_by_node = cur.execute(q).fetchall()

        now = time.time()
        rows_out: List[Dict[str, Any]] = []

        # reference offset
        ref_offset_ms: Optional[float] = None
        for r in last_by_node:
            if str(row_get(r, "node_id", "")) == str(reference_node):
                off = _f(row_get(r, "offset"))
                if off is not None:
                    ref_offset_ms = off * 1000.0

        for r in last_by_node:
            node_id = str(row_get(r, "node_id", ""))

            off = _f(row_get(r, "offset"))
            offset_ms = (off * 1000.0) if off is not None else None

            created_at = _f(row_get(r, "created_at"))
            age_s = (now - created_at) if created_at is not None else None

            t_wall = _f(row_get(r, "t_wall"))

            delta_vs_ref_ms = None
            if offset_ms is not None and ref_offset_ms is not None:
                delta_vs_ref_ms = offset_ms - ref_offset_ms

            rows_out.append({
                "node_id": node_id,
                "created_at": created_at,
                "created_at_utc": _utc(created_at),
                "offset_ms": offset_ms,
                "age_s": age_s,
                "delta_vs_ref_ms": delta_vs_ref_ms,
                "t_wall": t_wall,
                "t_wall_utc": _utc(t_wall),
            })

        return {"rows": rows_out, "reference_node": reference_node}
    except Exception:
        return {"rows": [], "reference_node": reference_node}
    finally:
        try:
            conn.close()
        except Exception:
            pass


# ------------------------------------------------------------
# Bin-synced timeseries (created_at axis)
# ------------------------------------------------------------
def get_ntp_timeseries(
    window_seconds: int = WINDOW_SECONDS_DEFAULT,
    max_points: int = MAX_POINTS_DEFAULT,
    bin_s: float = BIN_S_DEFAULT
) -> Dict[str, Any]:
    rows = fetch_node_rows(window_s=float(window_seconds), limit=int(max_points))
    if not rows:
        return {
            "series": {}, "pairs": {}, "jitter": {},
            "heatmap": {"data": [], "n_bins": 0},
            "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"},
        }

    if bin_s <= 0:
        bin_s = BIN_S_DEFAULT

    t_list: List[float] = []
    for r in rows:
        t = _f(row_get(r, "created_at"))
        if t is not None:
            t_list.append(t)

    if not t_list:
        return {
            "series": {}, "pairs": {}, "jitter": {},
            "heatmap": {"data": [], "n_bins": 0},
            "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"},
        }

    t_min = min(t_list)

    # bins[idx][node] = (created_at, offset_ms)
    bins: Dict[int, Dict[str, Tuple[float, float]]] = {}

    for r in rows:
        node = str(row_get(r, "node_id", "") or "")
        t = _f(row_get(r, "created_at"))
        off = _f(row_get(r, "offset"))
        if not node or t is None or off is None:
            continue
        offset_ms = off * 1000.0

        idx = int((t - t_min) / bin_s)
        if idx < 0:
            continue
        bucket = bins.setdefault(idx, {})
        prev = bucket.get(node)
        if (prev is None) or (t >= prev[0]):
            bucket[node] = (t, offset_ms)

    # per-bin center around mean (consensus view)
    for idx, bucket in bins.items():
        if len(bucket) < 2:
            continue
        mean = sum(off for (_t, off) in bucket.values()) / len(bucket)
        for node, (t_last, off_ms) in list(bucket.items()):
            bucket[node] = (t_last, off_ms - mean)

    # per-node series
    series: Dict[str, List[Dict[str, float]]] = {}
    for idx in sorted(bins.keys()):
        bucket = bins[idx]
        t_bin = t_min + (idx + 0.5) * bin_s
        for node, (_t, off_ms) in bucket.items():
            series.setdefault(node, []).append({"t_wall": t_bin, "offset_ms": off_ms})

    # delta per node
    for node, pts in series.items():
        pts.sort(key=lambda p: p["t_wall"])
        prev = None
        for p in pts:
            cur = p["offset_ms"]
            p["delta_offset_ms"] = 0.0 if prev is None else (cur - prev)
            prev = cur

    # pairwise deltas
    def norm_pair(a: str, b: str) -> str:
        return "-".join(sorted([a, b]))

    pairs: Dict[str, List[Dict[str, float]]] = {}
    for idx in sorted(bins.keys()):
        bucket = bins[idx]
        nodes_now = sorted(bucket.keys())
        if len(nodes_now) < 2:
            continue
        t_bin = t_min + (idx + 0.5) * bin_s
        off_map = {n: bucket[n][1] for n in nodes_now}

        for i in range(len(nodes_now)):
            for j in range(i + 1, len(nodes_now)):
                a = nodes_now[i]
                b = nodes_now[j]
                pair_id = norm_pair(a, b)
                delta_ms = off_map[a] - off_map[b]
                pairs.setdefault(pair_id, []).append({"t_wall": t_bin, "delta_ms": float(delta_ms), "bin": float(idx)})

    # mean-center per pair (visual)
    for pair_id, pts in pairs.items():
        if not pts:
            continue
        m = sum(float(p["delta_ms"]) for p in pts) / len(pts)
        for p in pts:
            p["delta_ms"] = float(p["delta_ms"]) - m

    # robust jitter per pair
    jitter: Dict[str, Dict[str, float]] = {}
    for pair_id, pts in pairs.items():
        if len(pts) < JITTER_MIN_SAMPLES:
            continue
        deltas = [float(p["delta_ms"]) for p in pts]
        iqr = robust_iqr(deltas)
        mad = robust_mad(deltas)
        sigma = 0.7413 * iqr
        jitter[pair_id] = {"sigma_ms": float(sigma), "iqr_ms": float(iqr), "mad_ms": float(mad), "n": float(len(deltas))}

    # heatmap (binned)
    heatmap_data: List[Dict[str, Any]] = []
    if pairs and bins:
        idx_min = min(bins.keys())
        idx_max = max(bins.keys())
        total_bins = max(1, idx_max - idx_min + 1)
        display_bins = min(HEATMAP_MAX_BINS, total_bins)

        if display_bins == total_bins:
            for pair_id, pts in pairs.items():
                for p in pts:
                    heatmap_data.append({"pair": pair_id, "t_bin": float(p["t_wall"]), "value": abs(float(p["delta_ms"]))})
            n_bins = total_bins
        else:
            accum: Dict[Tuple[str, int], List[float]] = {}
            for pair_id, pts in pairs.items():
                for p in pts:
                    bi = int(p.get("bin", idx_min))
                    bi = max(idx_min, min(idx_max, bi))
                    rel = (bi - idx_min) / max(1.0, float(total_bins))
                    d_idx = int(rel * display_bins)
                    if d_idx >= display_bins:
                        d_idx = display_bins - 1
                    accum.setdefault((pair_id, d_idx), []).append(abs(float(p["delta_ms"])))

            for (pair_id, d_idx), vals in accum.items():
                idx_center = idx_min + (d_idx + 0.5) * (total_bins / display_bins)
                t_center = t_min + (idx_center + 0.5) * bin_s
                heatmap_data.append({"pair": pair_id, "t_bin": float(t_center), "value": sum(vals) / max(1, len(vals))})

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
# Link-metrics timeseries (created_at axis)
# ------------------------------------------------------------
def get_link_timeseries(
    window_seconds: int = WINDOW_SECONDS_DEFAULT,
    max_points: int = LINK_MAX_POINTS_DEFAULT,
    bin_s: float = BIN_S_DEFAULT
) -> Dict[str, Any]:
    rows = fetch_link_rows(window_s=float(window_seconds), limit=int(max_points))
    if not rows:
        return {"links": {}, "latest_sigma": {}, "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"}}

    if bin_s <= 0:
        bin_s = BIN_S_DEFAULT

    t_list: List[float] = []
    for r in rows:
        t = _f(row_get(r, "created_at"))
        if t is not None:
            t_list.append(t)
    if not t_list:
        return {"links": {}, "latest_sigma": {}, "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"}}

    t_min = min(t_list)

    def lid(node_id: str, peer_id: str) -> str:
        return f"{node_id}->{peer_id}"

    # bins[idx][link] = (t, theta, rtt, sigma)
    bins: Dict[int, Dict[str, Tuple[float, Optional[float], Optional[float], Optional[float]]]] = {}

    for r in rows:
        t = _f(row_get(r, "created_at"))
        if t is None:
            continue
        nid = str(row_get(r, "node_id", "") or "")
        pid = str(row_get(r, "peer_id", "") or "")
        if not nid or not pid:
            continue

        link_id = lid(nid, pid)
        theta_ms = _f(row_get(r, "theta_ms"))
        rtt_ms = _f(row_get(r, "rtt_ms"))
        sigma_ms = _f(row_get(r, "sigma_ms"))

        idx = int((t - t_min) / bin_s)
        if idx < 0:
            continue

        bucket = bins.setdefault(idx, {})
        prev = bucket.get(link_id)
        if (prev is None) or (t >= prev[0]):
            bucket[link_id] = (t, theta_ms, rtt_ms, sigma_ms)

    links: Dict[str, List[Dict[str, Any]]] = {}
    latest_sigma: Dict[str, float] = {}

    for idx in sorted(bins.keys()):
        bucket = bins[idx]
        t_bin = t_min + (idx + 0.5) * bin_s
        for link_id, (_t, theta_ms, rtt_ms, sigma_ms) in bucket.items():
            obj: Dict[str, Any] = {"t_wall": t_bin}
            if theta_ms is not None:
                obj["theta_ms"] = theta_ms
            if rtt_ms is not None:
                obj["rtt_ms"] = rtt_ms
            if sigma_ms is not None:
                obj["sigma_ms"] = sigma_ms
                latest_sigma[link_id] = sigma_ms
            links.setdefault(link_id, []).append(obj)

    for link_id, pts in links.items():
        pts.sort(key=lambda p: p["t_wall"])

    return {"links": links, "latest_sigma": latest_sigma, "meta": {"bin_s": bin_s, "window_seconds": window_seconds, "x_axis": "created_at"}}


# ------------------------------------------------------------
# Controller debug timeseries (node-only)
# ------------------------------------------------------------
def get_controller_timeseries(window_seconds: int = WINDOW_SECONDS_DEFAULT, max_points: int = 8000) -> Dict[str, Any]:
    rows = fetch_node_rows(window_s=float(window_seconds), limit=int(max_points))
    by_node: Dict[str, List[Dict[str, Any]]] = {}

    for r in rows:
        nid = str(row_get(r, "node_id", "") or "")
        t = _f(row_get(r, "created_at"))
        if not nid or t is None:
            continue

        obj: Dict[str, Any] = {"t_wall": t}

        for k in ["delta_desired_ms", "delta_applied_ms", "dt_s", "slew_clipped"]:
            if not row_has(r, k):
                continue
            v = row_get(r, k)
            if v is None:
                continue

            if k == "slew_clipped":
                try:
                    obj[k] = 1 if int(v) else 0
                except Exception:
                    pass
            else:
                fv = _f(v)
                if fv is not None:
                    obj[k] = fv

        by_node.setdefault(nid, []).append(obj)

    for nid, pts in by_node.items():
        pts.sort(key=lambda p: p["t_wall"])

    return {"controller": by_node, "meta": {"window_seconds": window_seconds, "x_axis": "created_at"}}


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
        # prefer C, else A, else first real node id
        ref = "C" if "C" in cfg else ("A" if "A" in cfg else None)
        if ref is None:
            for k in sorted(cfg.keys()):
                if k != "sync":
                    ref = k
                    break
        if ref is None:
            ref = "C"
        return jsonify(get_status_snapshot(reference_node=str(ref)))
    except Exception as e:
        return _json_error(f"/api/status failed: {e}", 500)


@app.route("/api/overview")
def api_overview():
    try:
        cfg = load_config()
        sync_cfg = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync_cfg.get("ui_window_s", WINDOW_SECONDS_DEFAULT))
        return jsonify(compute_overview(window_s))
    except Exception as e:
        return _json_error(f"/api/overview failed: {e}", 500)


@app.route("/api/ntp_timeseries")
def api_ntp_timeseries():
    try:
        cfg = load_config()
        sync_cfg = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync_cfg.get("ui_window_s", WINDOW_SECONDS_DEFAULT))
        bin_s = float(sync_cfg.get("ui_bin_s", BIN_S_DEFAULT))
        max_points = int(sync_cfg.get("ui_max_points", MAX_POINTS_DEFAULT))
        return jsonify(get_ntp_timeseries(window_seconds=int(window_s), max_points=max_points, bin_s=bin_s))
    except Exception as e:
        return _json_error(f"/api/ntp_timeseries failed: {e}", 500)


@app.route("/api/link_timeseries")
def api_link_timeseries():
    try:
        cfg = load_config()
        sync_cfg = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync_cfg.get("ui_window_s", WINDOW_SECONDS_DEFAULT))
        bin_s = float(sync_cfg.get("ui_bin_s", BIN_S_DEFAULT))
        max_points = int(sync_cfg.get("ui_link_max_points", LINK_MAX_POINTS_DEFAULT))
        return jsonify(get_link_timeseries(window_seconds=int(window_s), max_points=max_points, bin_s=bin_s))
    except Exception as e:
        return _json_error(f"/api/link_timeseries failed: {e}", 500)


@app.route("/api/controller_timeseries")
def api_controller_timeseries():
    try:
        cfg = load_config()
        sync_cfg = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync_cfg.get("ui_window_s", WINDOW_SECONDS_DEFAULT))
        max_points = int(sync_cfg.get("ui_ctrl_max_points", 8000))
        return jsonify(get_controller_timeseries(window_seconds=int(window_s), max_points=max_points))
    except Exception as e:
        return _json_error(f"/api/controller_timeseries failed: {e}", 500)


@app.template_filter("datetime_utc")
def datetime_utc(ts):
    return _utc(_f(ts))


# ------------------------------------------------------------
# Template
# ------------------------------------------------------------
TEMPLATE = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>MeshTime Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:0; padding: 1.5rem; background:#111; color:#eee; }
    h1,h2,h3 { margin:0; }
    .sub { margin-top:0.35rem; opacity:0.7; font-size:0.85rem; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .small { font-size:0.85rem; opacity:0.75; }
    .grid { display:grid; grid-template-columns: minmax(0,1.25fr) minmax(0,2fr); gap: 1.25rem; align-items:start; }
    @media (max-width: 1200px) { .grid { grid-template-columns: minmax(0,1fr); } }
    .card { background:#1b1b1b; border-radius:14px; padding:1rem 1.25rem; box-shadow: 0 0 0 1px rgba(255,255,255,0.05); }
    .row { display:flex; gap: 0.75rem; flex-wrap:wrap; align-items:center; }
    .pill { display:inline-block; padding:0.12rem 0.55rem; border-radius:999px; font-size:0.75rem; font-weight:650; }
    .ok   { background: rgba(46,204,113,0.14); color:#2ecc71; }
    .warn { background: rgba(241,196,15,0.14); color:#f1c40f; }
    .bad  { background: rgba(231,76,60,0.14); color:#e74c3c; }
    .muted { opacity:0.7; }

    .kpi-grid { display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 0.75rem; margin-top: 0.75rem; }
    @media (max-width: 900px) { .kpi-grid { grid-template-columns: minmax(0,1fr); } }
    .kpi { border-radius: 12px; padding: 0.75rem; background: rgba(255,255,255,0.03); box-shadow: 0 0 0 1px rgba(255,255,255,0.03); }
    .kpi .title { font-size:0.85rem; opacity:0.75; }
    .kpi .value { font-size:1.25rem; font-weight:700; margin-top:0.2rem; }
    .kpi .hint { margin-top:0.25rem; font-size:0.8rem; opacity:0.65; }

    table { width:100%; border-collapse: collapse; font-size:0.9rem; margin-top: 0.6rem; }
    th, td { padding: 0.35rem 0.5rem; text-align:left; vertical-align:top; }
    th { border-bottom: 1px solid rgba(255,255,255,0.15); font-weight:650; }
    tr:nth-child(even) td { background: rgba(255,255,255,0.02); }

    canvas { max-width:100%; }
    .plots { display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; margin-top:0.75rem; }
    @media (max-width: 1200px) { .plots { grid-template-columns: minmax(0,1fr); } }

    #meshCanvas { width:100%; height: 260px; background:#171717; border-radius:12px; margin-top:0.6rem; }

    .toggle { display:flex; gap:0.6rem; align-items:center; margin-top: 0.75rem; }
    .toggle input { transform: scale(1.2); }
    .hidden { display:none !important; }
  </style>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@1.3.0/dist/chartjs-chart-matrix.min.js"></script>
</head>

<body>
  <div class="row" style="justify-content:space-between;">
    <div>
      <h1>MeshTime Dashboard</h1>
      <div class="sub">
        Datenquelle: <span class="mono">{{ db_path }}</span> · X-Achse: <b>created_at (Sink-Clock)</b>
      </div>
    </div>

    <div class="toggle card" style="padding:0.55rem 0.85rem;">
      <input id="debugToggle" type="checkbox">
      <label for="debugToggle" class="small">Debug-Mode (Controller)</label>
    </div>
  </div>

  <div class="grid" style="margin-top:1.25rem;">
    <div>
      <div class="card">
        <h2>Overview</h2>
        <div class="small" id="overviewMeta">lade…</div>

        <div class="kpi-grid" id="nodeKpis">
          <div class="kpi"><div class="title">Nodes</div><div class="value">…</div><div class="hint">—</div></div>
        </div>

        <div class="spacer" style="height:0.75rem;"></div>

        <h3 style="font-size:1rem; margin-top:0.5rem;">Offenders</h3>
        <div class="small" id="offendersLine">lade…</div>
      </div>

      <div class="card" style="margin-top:1rem;">
        <h2>Aktueller Status (node-only)</h2>
        <table id="statusTable">
          <thead>
            <tr>
              <th>Node</th>
              <th>Last Seen (UTC)</th>
              <th>Δ vs Ref</th>
              <th>Offset</th>
              <th>Age</th>
              <th>Sender Wallclock</th>
            </tr>
          </thead>
          <tbody><tr><td colspan="6" class="small">lade…</td></tr></tbody>
        </table>
        <div class="small" id="statusMeta"></div>
      </div>

      <div class="card" style="margin-top:1rem;">
        <h2>Topologie</h2>
        <canvas id="meshCanvas"></canvas>
        <div class="small">Grün = Node hat frische Daten; gelb = warmup; rot = stale/noise. Dicker Rand = Root.</div>
      </div>
    </div>

    <div>
      <div class="card">
        <h2>Offsets & Konsens</h2>
        <div class="plots">
          <div>
            <h3 class="small">1) Paarweise ΔOffset (ms)</h3>
            <canvas id="pairChart" height="160"></canvas>
          </div>
          <div>
            <h3 class="small">2) Δoffset pro Node (ms)</h3>
            <canvas id="offsetDeltaChart" height="160"></canvas>
          </div>
          <div>
            <h3 class="small">3) Jitter (robust σ ≈ 0.741·IQR)</h3>
            <canvas id="jitterBarChart" height="160"></canvas>
          </div>
          <div>
            <h3 class="small">4) Heatmap |Δ| (gebinnt)</h3>
            <canvas id="heatmapChart" height="160"></canvas>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top:1rem;">
        <h2>Links (θ/RTT/σ)</h2>
        <div class="small muted">Link-ID ist <span class="mono">Node-&gt;Peer</span> (gerichtet; θ ist signiert).</div>

        <div class="plots">
          <div>
            <h3 class="small">5) θ pro Link (ms)</h3>
            <canvas id="thetaChart" height="160"></canvas>
          </div>
          <div>
            <h3 class="small">6) RTT pro Link (ms)</h3>
            <canvas id="rttChart" height="160"></canvas>
          </div>
          <div>
            <h3 class="small">7) Link σ (latest)</h3>
            <canvas id="linkSigmaBarChart" height="160"></canvas>
          </div>
          <div class="small">
            <b>Interpretation:</b><br>
            θ stabil, ΔOffset driftet → Controller/Bias/Offset-Schätzung.<br>
            RTT spiky → Queueing/Medium. σ hoch → Jitter/Asymmetrie/Noise.
          </div>
        </div>
      </div>

      <div class="card debugOnly" style="margin-top:1rem;">
        <h2>Controller (Debug)</h2>
        <div class="small muted">delta_desired vs delta_applied, dt, slew_clipped</div>

        <div class="plots">
          <div>
            <h3 class="small">8) delta_desired_ms</h3>
            <canvas id="deltaDesiredChart" height="160"></canvas>
          </div>
          <div>
            <h3 class="small">9) delta_applied_ms</h3>
            <canvas id="deltaAppliedChart" height="160"></canvas>
          </div>
          <div>
            <h3 class="small">10) dt_s</h3>
            <canvas id="dtChart" height="160"></canvas>
          </div>
          <div>
            <h3 class="small">11) slew_clipped (0/1)</h3>
            <canvas id="slewChart" height="160"></canvas>
          </div>
        </div>
      </div>

      <div class="small muted" style="margin-top:1rem;">
        Reality check: wenn Konvergenz “fühlt” aber UI sagt nein → schau Debug: slew_clipped? link_sigma? freshness?
      </div>
    </div>
  </div>

<script>
  const colors = [
    'rgba(46, 204, 113, 0.55)',
    'rgba(52, 152, 219, 0.55)',
    'rgba(231, 76, 60, 0.55)',
    'rgba(241, 196, 15, 0.55)',
    'rgba(155, 89, 182, 0.55)',
    'rgba(26, 188, 156, 0.55)',
    'rgba(230, 126, 34, 0.55)',
  ];

  let pairChart, offsetDeltaChart, jitterBarChart, heatmapChart;
  let thetaChart, rttChart, linkSigmaBarChart;
  let deltaDesiredChart, deltaAppliedChart, dtChart, slewChart;

  function lineChart(ctx, yLabel, yFormatFn=null) {
    return new Chart(ctx, {
      type: 'line',
      data: { datasets: [] },
      options: {
        responsive: true,
        animation: false,
        scales: {
          x: { type:'time', time:{ unit:'second' }, ticks:{ color:'#aaa' }, grid:{ color:'rgba(255,255,255,0.06)' } },
          y: {
            ticks:{
              color:'#aaa',
              callback:(v)=> yFormatFn ? yFormatFn(v) : ((v.toFixed ? v.toFixed(2) : v) + ' ' + yLabel)
            },
            grid:{ color:'rgba(255,255,255,0.06)' }
          }
        },
        plugins: {
          legend: { labels:{ color:'#eee' } },
          tooltip: { callbacks: { label:(ctx)=> `${ctx.dataset.label}: ${ctx.parsed.y}` } }
        },
        elements: { line:{ tension:0.1 }, point:{ radius:0 } }
      }
    });
  }

  function barChart(ctx, label, yLabel) {
    return new Chart(ctx, {
      type:'bar',
      data:{ labels:[], datasets:[{ label:label, data:[], backgroundColor:'rgba(52,152,219,0.75)' }] },
      options:{
        responsive:true,
        animation:false,
        scales:{
          x:{ ticks:{ color:'#aaa' }, grid:{ display:false } },
          y:{ ticks:{ color:'#aaa', callback:(v)=> (v.toFixed? v.toFixed(2):v) + ' ' + yLabel }, grid:{ color:'rgba(255,255,255,0.06)' } }
        },
        plugins:{ legend:{ labels:{ color:'#eee' } } }
      }
    });
  }

  function heatmapChartFactory(ctx) {
    return new Chart(ctx, {
      type: 'matrix',
      data: { datasets: [{
        label: 'Heatmap',
        data: [],
        borderWidth: 0,
        backgroundColor: (context) => {
          const chart = context.chart;
          const raw = context.raw;
          if (!raw || typeof raw.v !== 'number') return 'rgba(0,0,0,0)';
          const maxV = chart._maxV || 1;
          const ratio = Math.min(1, raw.v / maxV);
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
      }]},
      options: {
        responsive:true,
        animation:false,
        scales: {
          x: { type:'time', time:{ unit:'second' }, ticks:{ color:'#aaa' }, grid:{ color:'rgba(255,255,255,0.06)' } },
          y: {
            type:'linear',
            ticks:{
              color:'#aaa',
              callback: function(value){
                const cats = this.chart._categories || [];
                const idx = Math.round(value);
                return cats[idx] || '';
              }
            },
            grid:{ color:'rgba(255,255,255,0.06)' }
          }
        },
        plugins:{
          legend:{ labels:{ color:'#eee' } },
          tooltip:{ callbacks:{ label:(ctx)=>{
            const cats = ctx.chart._categories || [];
            const pairId = cats[Math.round(ctx.raw.y)] || '?';
            const v = (ctx.raw && typeof ctx.raw.v === 'number') ? ctx.raw.v : 0;
            const t = new Date(ctx.raw.x);
            return `${pairId} @ ${t.toLocaleTimeString()} : |Δ|=${v.toFixed(2)} ms`;
          }}}
        }
      }
    });
  }

  function initCharts(){
    pairChart = lineChart(document.getElementById('pairChart').getContext('2d'), 'ms');
    offsetDeltaChart = lineChart(document.getElementById('offsetDeltaChart').getContext('2d'), 'ms');
    jitterBarChart = barChart(document.getElementById('jitterBarChart').getContext('2d'), 'robust σ', 'ms');
    heatmapChart = heatmapChartFactory(document.getElementById('heatmapChart').getContext('2d'));

    thetaChart = lineChart(document.getElementById('thetaChart').getContext('2d'), 'ms');
    rttChart = lineChart(document.getElementById('rttChart').getContext('2d'), 'ms');
    linkSigmaBarChart = barChart(document.getElementById('linkSigmaBarChart').getContext('2d'), 'σ (latest)', 'ms');

    deltaDesiredChart = lineChart(document.getElementById('deltaDesiredChart').getContext('2d'), 'ms');
    deltaAppliedChart = lineChart(document.getElementById('deltaAppliedChart').getContext('2d'), 'ms');
    dtChart = lineChart(document.getElementById('dtChart').getContext('2d'), 's');
    slewChart = lineChart(document.getElementById('slewChart').getContext('2d'), '', (v)=> v);
  }

  function updateLineChart(chart, seriesMap, field, labelPrefix=''){
    const ids = Object.keys(seriesMap||{}).sort();
    chart.data.datasets = [];
    ids.forEach((id, idx)=>{
      const base = colors[idx % colors.length];
      const stroke = base.replace('0.55','0.95');
      const pts = (seriesMap[id]||[])
        .filter(p => p && p[field] !== null && p[field] !== undefined && Number.isFinite(p[field]))
        .map(p => ({ x: new Date(p.t_wall*1000), y: p[field] }));
      chart.data.datasets.push({ label: `${labelPrefix}${id}`, data: pts, borderColor: stroke, backgroundColor: base, fill:false, pointRadius:0, borderWidth:1.5 });
    });
    chart.update();
  }

  function updatePairs(chart, pairs){
    const ids = Object.keys(pairs||{}).sort();
    chart.data.datasets = [];
    ids.forEach((id, idx)=>{
      const base = colors[idx % colors.length];
      const stroke = base.replace('0.55','0.95');
      const pts = (pairs[id]||[]).map(p => ({ x: new Date(p.t_wall*1000), y: p.delta_ms }));
      chart.data.datasets.push({ label:id, data:pts, borderColor:stroke, backgroundColor:base, fill:false, pointRadius:0, borderWidth:1.5 });
    });
    chart.update();
  }

  function updateJitter(chart, jitter){
    const ids = Object.keys(jitter||{}).sort();
    const labels=[], data=[];
    ids.forEach(id=>{
      const s = jitter[id]?.sigma_ms;
      if (s !== null && s !== undefined && Number.isFinite(s)) { labels.push(id); data.push(s); }
    });
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update();
  }

  function updateHeatmap(chart, heatmap){
    const data = (heatmap && heatmap.data) || [];
    if (!data.length){
      chart.data.datasets[0].data = [];
      chart._categories = [];
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
      if (v > maxV) maxV = v;
      return { x: new Date(d.t_bin*1000), y: idx.get(d.pair) ?? 0, v: v };
    });
    chart.data.datasets[0].data = matrix;
    chart._categories = cats;
    chart._nBins = heatmap.n_bins || 1;
    chart._maxV = maxV || 1;
    chart.update();
  }

  function updateLinkSigmaBar(chart, latest){
    const ids = Object.keys(latest||{}).sort();
    const labels=[], data=[];
    ids.forEach(id=>{
      const v = latest[id];
      if (v !== null && v !== undefined && Number.isFinite(v)) { labels.push(id); data.push(v); }
    });
    chart.data.labels = labels;
    chart.data.datasets[0].data = data;
    chart.update();
  }

  async function fetchJson(url){
    const resp = await fetch(url);
    let data = null;
    try { data = await resp.json(); } catch(e) { data = { error: "invalid json" }; }
    if (!resp.ok) throw new Error(`${url}: ${data.error || 'unknown error'}`);
    return data;
  }

  function pillClass(state){
    if (state === 'GREEN') return 'pill ok';
    if (state === 'RED') return 'pill bad';
    return 'pill warn';
  }

  function renderOverview(ov){
    const meta = ov?.meta || {};
    const nodes = ov?.nodes || [];
    const offenders = ov?.offenders || {};

    const thr = meta.thresholds || {};
    const freshS = (thr.fresh_s !== null && thr.fresh_s !== undefined) ? thr.fresh_s : '?';
    const dmed   = (thr.delta_applied_med_ms !== null && thr.delta_applied_med_ms !== undefined) ? thr.delta_applied_med_ms : '?';
    const clip   = (thr.slew_clip_rate !== null && thr.slew_clip_rate !== undefined) ? Math.round(thr.slew_clip_rate*100) : '?';
    const lsig   = (thr.link_sigma_med_ms !== null && thr.link_sigma_med_ms !== undefined) ? thr.link_sigma_med_ms : '?';

    document.getElementById('overviewMeta').textContent =
      `now=${meta.now_utc || 'n/a'} · conv_window=${meta.conv_window_s || '?'}s · thresholds: fresh≤${freshS}s, |Δapplied|med≤${dmed}ms, clip≤${clip}%, linkσmed≤${lsig}ms`;

    const kpiWrap = document.getElementById('nodeKpis');
    kpiWrap.innerHTML = '';

    nodes.forEach(n=>{
      const off = (n.offset_centered_ms !== null && n.offset_centered_ms !== undefined) ? n.offset_centered_ms.toFixed(2)+' ms' : 'n/a';
      const age = (n.age_s !== null && n.age_s !== undefined) ? n.age_s.toFixed(1)+' s' : 'n/a';
      const da  = (n.med_abs_delta_applied_ms !== null && n.med_abs_delta_applied_ms !== undefined) ? n.med_abs_delta_applied_ms.toFixed(2)+' ms' : 'n/a';
      const sig = (n.link_sigma_med_ms !== null && n.link_sigma_med_ms !== undefined) ? n.link_sigma_med_ms.toFixed(2)+' ms' : '—';
      const clipr = (n.slew_clip_rate !== null && n.slew_clip_rate !== undefined) ? Math.round(n.slew_clip_rate*100)+'%' : '—';

      const div = document.createElement('div');
      div.className = 'kpi';
      div.innerHTML = `
        <div class="row" style="justify-content:space-between;">
          <div class="title">Node ${n.node_id}</div>
          <span class="${pillClass(n.state)}">${n.state}</span>
        </div>
        <div class="value">${off}</div>
        <div class="hint">age ${age} · |Δapplied|med ${da} · clip ${clipr} · linkσmed ${sig}</div>
        <div class="small muted">${n.reason || ''}</div>
      `;
      kpiWrap.appendChild(div);
    });

    document.getElementById('offendersLine').textContent =
      `worst |Δapplied|: ${offenders.worst_node_delta || '—'} · stalest: ${offenders.worst_node_freshness || '—'} · most clipped: ${offenders.most_slew_clipped || '—'} · worst link σ: ${offenders.worst_link_sigma || '—'}`;
  }

  function classifyDelta(absMs){
    if (absMs < 5) return 'pill ok';
    if (absMs < 20) return 'pill warn';
    return 'pill bad';
  }

  function renderStatusTable(status){
    const tbody = document.querySelector('#statusTable tbody');
    tbody.innerHTML = '';
    const rows = (status && status.rows) || [];
    const ref = status && status.reference_node ? status.reference_node : '?';
    document.getElementById('statusMeta').textContent = rows.length ? `Referenz für Δ: Node ${ref} (Δ aus Offsets; node-only rows)` : '';

    if (!rows.length){
      const tr = document.createElement('tr');
      tr.innerHTML = `<td colspan="6" class="small">Keine ntp_reference(node-only) Daten.</td>`;
      tbody.appendChild(tr);
      return;
    }

    rows.forEach(r=>{
      const node = r.node_id;
      const lastSeenUTC = r.created_at_utc || 'n/a';
      const off = (r.offset_ms !== null && r.offset_ms !== undefined) ? r.offset_ms : null;
      const age = (r.age_s !== null && r.age_s !== undefined) ? r.age_s : null;
      const dRef = (r.delta_vs_ref_ms !== null && r.delta_vs_ref_ms !== undefined) ? r.delta_vs_ref_ms : null;
      const tWallUTC = r.t_wall_utc || 'n/a';

      const dcls = (dRef === null) ? 'pill warn' : classifyDelta(Math.abs(dRef));
      const ocls = (off === null) ? 'pill warn' : classifyDelta(Math.abs(off));

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${node}</td>
        <td>${lastSeenUTC}</td>
        <td>${dRef === null ? '<span class="small">n/a</span>' : `<span class="${dcls}">${dRef.toFixed(2)} ms</span>`}</td>
        <td>${off === null ? '<span class="small">n/a</span>' : `<span class="${ocls}">${off.toFixed(2)} ms</span>`}</td>
        <td>${age === null ? '<span class="small">n/a</span>' : `<span class="small">${age.toFixed(1)} s</span>`}</td>
        <td class="small">${tWallUTC}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function drawMesh(canvas, topo, nodeStates){
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0,0,w,h);

    const nodes = topo.nodes || [];
    const links = topo.links || [];
    if (!nodes.length){
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
      pos[node.id] = { x: cx + radius*Math.cos(a), y: cy + radius*Math.sin(a) };
    });

    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.lineWidth = 1.0;
    links.forEach(l=>{
      const a = pos[l.source], b = pos[l.target];
      if (!a || !b) return;
      ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
    });

    nodes.forEach(node=>{
      const p = pos[node.id]; if (!p) return;
      const st = nodeStates[node.id] || 'YELLOW';
      const isRoot = !!node.is_root;
      const r = isRoot ? 20 : 16;

      let fill = 'rgba(241,196,15,0.14)';
      let stroke = '#f1c40f';
      if (st === 'GREEN'){ fill='rgba(46,204,113,0.18)'; stroke='#2ecc71'; }
      if (st === 'RED'){ fill='rgba(231,76,60,0.18)'; stroke='#e74c3c'; }

      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,2*Math.PI);
      ctx.fillStyle = fill; ctx.fill();

      ctx.lineWidth = isRoot ? 3.0 : 2.0;
      ctx.strokeStyle = stroke; ctx.stroke();

      ctx.fillStyle = '#ecf0f1';
      ctx.font = '12px system-ui';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(node.id, p.x, p.y);
    });

    const roots = nodes.filter(n=>n.is_root);
    ctx.fillStyle = '#aaa';
    ctx.font = '11px system-ui';
    ctx.textAlign='left'; ctx.textBaseline='top';
    ctx.fillText(roots.length ? ('Root: ' + roots.map(r=>r.id).join(', ')) : 'Root: keiner', 10, 10);
  }

  function applyDebugVisibility(){
    const on = document.getElementById('debugToggle').checked;
    document.querySelectorAll('.debugOnly').forEach(el=>{
      el.classList.toggle('hidden', !on);
    });
  }

  async function refresh(){
    try{
      const [ov, ntp, link, status, topo, ctrl] = await Promise.all([
        fetchJson('/api/overview'),
        fetchJson('/api/ntp_timeseries'),
        fetchJson('/api/link_timeseries'),
        fetchJson('/api/status'),
        fetchJson('/api/topology'),
        fetchJson('/api/controller_timeseries'),
      ]);

      renderOverview(ov);
      renderStatusTable(status);

      updatePairs(pairChart, ntp.pairs || {});
      updateLineChart(offsetDeltaChart, ntp.series || {}, "delta_offset_ms", "Node ");
      updateJitter(jitterBarChart, ntp.jitter || {});
      updateHeatmap(heatmapChart, ntp.heatmap || {data:[], n_bins:0});

      const links = link.links || {};
      updateLineChart(thetaChart, links, "theta_ms", "");
      updateLineChart(rttChart, links, "rtt_ms", "");
      updateLinkSigmaBar(linkSigmaBarChart, link.latest_sigma || {});

      const c = ctrl.controller || {};
      updateLineChart(deltaDesiredChart, c, "delta_desired_ms", "Node ");
      updateLineChart(deltaAppliedChart, c, "delta_applied_ms", "Node ");
      updateLineChart(dtChart, c, "dt_s", "Node ");
      updateLineChart(slewChart, c, "slew_clipped", "Node ");

      const nodeStates = {};
      (ov.nodes || []).forEach(n => { nodeStates[n.node_id] = n.state; });

      const meshCanvas = document.getElementById('meshCanvas');
      meshCanvas.width = meshCanvas.clientWidth;
      meshCanvas.height = meshCanvas.clientHeight;
      drawMesh(meshCanvas, topo, nodeStates);

    } catch(e){
      console.error('refresh failed:', e);
    }
  }

  window.addEventListener('load', ()=>{
    initCharts();
    document.getElementById('debugToggle').addEventListener('change', ()=>{
      applyDebugVisibility();
    });
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
    Storage(str(DB_PATH))  # create schema if missing


if __name__ == "__main__":
    ensure_db()
    print("Starting MeshTime Web-UI on http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)
