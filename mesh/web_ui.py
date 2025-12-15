# -*- coding: utf-8 -*-
# mesh/web_ui.py
#
# MeshTime Dashboard (Vision-First Rewrite)
#
# Principles:
#   - The ONE axis of truth in an unrooted mesh UI is created_at (sink clock).
#   - Mesh-Time is first-class: show t_mesh(node) (or its best available proxy) as the primary view.
#   - Deltas are diagnosis, not "truth".
#   - Convergence is computed deterministically from data: correction small, links stable, data fresh (+ warmup state).
#   - Two modes:
#       Operator-Mode: minimal, unambiguous, fast scan.
#       Debug-Mode: controller internals + extra charts.
#   - HARD RULE: this server must NEVER crash due to None / missing columns / formatting.
#
# DB contract (ntp_reference):
#   - node-only samples: peer_id IS NULL
#   - link samples:     peer_id NOT NULL (theta_ms, rtt_ms, sigma_ms)
#
# Requires:
#   pip install flask
#
# Run:
#   python -m mesh.web_ui   (or python mesh/web_ui.py when PYTHONPATH is set)
#
from __future__ import annotations

import json
import math
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flask import Flask, jsonify, make_response, render_template_string, request

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "mesh_data.sqlite"
CFG_PATH = BASE_DIR / "config" / "nodes.json"

from mesh.storage import Storage  # ensures schema (safe to import)

app = Flask(__name__)

# -----------------------------
# Defaults / UI tuning
# -----------------------------
WINDOW_S_DEFAULT = 120.0
BIN_S_DEFAULT = 0.5

NODE_MAX_ROWS_DEFAULT = 20000
LINK_MAX_ROWS_DEFAULT = 30000

# Convergence tuning (Operator-mode)
CONV_WINDOW_S = 30.0

# These get adjusted with config when possible
THRESH_FRESH_MULT = 6.0              # fresh <= beacon_period_s * THRESH_FRESH_MULT
THRESH_FRESH_MIN_S = 3.0
THRESH_DELTA_APPLIED_MED_MS = 0.5
THRESH_SLEW_CLIP_RATE = 0.20
THRESH_LINK_SIGMA_MED_MS = 2.0

MIN_SAMPLES_WARMUP = 10              # proxy until beacon_count exists
MIN_LINK_SAMPLES_FOR_STABLE = 4      # "at least K" samples in conv window

HEATMAP_MAX_BINS = 48

# -----------------------------
# Safe helpers (never crash)
# -----------------------------
def _json_error(msg: str, status: int = 500):
    return make_response(jsonify({"error": str(msg)}), int(status))


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _utc(ts: Optional[float]) -> str:
    if ts is None:
        return "n/a"
    try:
        return datetime.utcfromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "n/a"


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


def _table_cols(conn: sqlite3.Connection, table: str) -> Set[str]:
    try:
        cur = conn.cursor()
        return {row[1] for row in cur.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def _select_existing(cols: Set[str], wanted: List[str]) -> List[str]:
    return [c for c in wanted if c in cols]


# -----------------------------
# Robust stats (fast + safe)
# -----------------------------
def _quantile_sorted(xs: List[float], q: float) -> float:
    n = len(xs)
    if n <= 0:
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
    return float(_quantile_sorted(xs, 0.50))


def robust_iqr(values: List[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    q1 = _quantile_sorted(xs, 0.25)
    q3 = _quantile_sorted(xs, 0.75)
    return float(q3 - q1)


def robust_mad(values: List[float]) -> Optional[float]:
    if not values:
        return None
    xs = sorted(values)
    med = _quantile_sorted(xs, 0.50)
    dev = sorted([abs(v - med) for v in values])
    return float(_quantile_sorted(dev, 0.50))


# -----------------------------
# Config / topology
# -----------------------------
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
    nodes: List[Dict[str, Any]] = []
    links: List[Dict[str, Any]] = []

    for node_id, entry in (cfg.items() if isinstance(cfg, dict) else []):
        if node_id == "sync":
            continue
        if not isinstance(entry, dict):
            continue

        ip = entry.get("ip")
        color = entry.get("color") or "#3498db"
        sync_cfg = entry.get("sync", {}) or {}
        is_root = bool(sync_cfg.get("is_root", False))
        neighs = entry.get("neighbors", []) or []

        nodes.append({"id": str(node_id), "ip": ip, "color": color, "is_root": is_root})
        for n in neighs:
            try:
                n = str(n)
            except Exception:
                continue
            # store undirected edges for display
            if node_id < n:
                links.append({"a": str(node_id), "b": n})

    return {"nodes": nodes, "links": links}


# -----------------------------
# DB fetch (node-only vs link rows)
# -----------------------------
def fetch_node_rows(window_s: float, limit: int) -> List[sqlite3.Row]:
    conn = get_conn()
    try:
        cols = _table_cols(conn, "ntp_reference")
        if "node_id" not in cols or "created_at" not in cols:
            return []

        wanted = [
            "id", "node_id", "created_at", "peer_id",
            "t_mesh", "t_wall", "t_mono", "offset",
            "delta_desired_ms", "delta_applied_ms", "dt_s", "slew_clipped",
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


# -----------------------------
# Derived quantities
# -----------------------------
def _proxy_mesh_time_ms(r: sqlite3.Row) -> Optional[float]:
    """
    Best available 'Mesh-Time' proxy (in ms) for a node-only row.
    Priority:
      1) t_mesh (seconds) -> ms
      2) t_mono + offset (seconds + seconds) -> ms
      3) offset alone (seconds) -> ms (weak, but better than nothing for centering)
    """
    t_mesh = _f(row_get(r, "t_mesh"))
    if t_mesh is not None:
        return t_mesh * 1000.0

    t_mono = _f(row_get(r, "t_mono"))
    off_s = _f(row_get(r, "offset"))
    if t_mono is not None and off_s is not None:
        return (t_mono + off_s) * 1000.0

    if off_s is not None:
        return off_s * 1000.0

    return None


def _offset_ms(r: sqlite3.Row) -> Optional[float]:
    off_s = _f(row_get(r, "offset"))
    return None if off_s is None else off_s * 1000.0


def _fresh_threshold_s_from_cfg(cfg: Dict[str, Any]) -> float:
    try:
        sync_cfg = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        beacon_period_s = float(sync_cfg.get("beacon_period_s", 0.5))
        dyn = max(THRESH_FRESH_MIN_S, beacon_period_s * THRESH_FRESH_MULT)
        return float(dyn)
    except Exception:
        return float(THRESH_FRESH_MIN_S)


# -----------------------------
# Aggregations (backend does the work)
# -----------------------------
@dataclass
class ConvThresholds:
    fresh_s: float
    delta_applied_med_ms: float
    slew_clip_rate: float
    link_sigma_med_ms: float
    conv_window_s: float
    min_samples_warmup: int
    min_link_samples: int


def compute_overview(window_s: float) -> Dict[str, Any]:
    cfg = load_config()
    now = time.time()

    eff_window = max(float(window_s), float(CONV_WINDOW_S))
    node_rows = fetch_node_rows(window_s=eff_window, limit=NODE_MAX_ROWS_DEFAULT)
    link_rows = fetch_link_rows(window_s=eff_window, limit=LINK_MAX_ROWS_DEFAULT)

    thr = ConvThresholds(
        fresh_s=_fresh_threshold_s_from_cfg(cfg),
        delta_applied_med_ms=THRESH_DELTA_APPLIED_MED_MS,
        slew_clip_rate=THRESH_SLEW_CLIP_RATE,
        link_sigma_med_ms=THRESH_LINK_SIGMA_MED_MS,
        conv_window_s=CONV_WINDOW_S,
        min_samples_warmup=MIN_SAMPLES_WARMUP,
        min_link_samples=MIN_LINK_SAMPLES_FOR_STABLE,
    )

    # latest node-only row per node (by created_at)
    latest: Dict[str, sqlite3.Row] = {}
    by_node_recent: Dict[str, List[sqlite3.Row]] = {}

    for r in node_rows:
        nid = str(row_get(r, "node_id", "") or "")
        if not nid:
            continue
        by_node_recent.setdefault(nid, []).append(r)
        latest[nid] = r

    # latest link row per directed link + all samples by directed link
    by_link: Dict[str, List[sqlite3.Row]] = {}
    for r in link_rows:
        nid = str(row_get(r, "node_id", "") or "")
        pid = str(row_get(r, "peer_id", "") or "")
        if not nid or not pid:
            continue
        lid = f"{nid}->{pid}"
        by_link.setdefault(lid, []).append(r)

    # mesh median at "now" from latest proxy mesh time values
    latest_mesh_ms: List[float] = []
    for r in latest.values():
        tm = _proxy_mesh_time_ms(r)
        if tm is not None:
            latest_mesh_ms.append(tm)
    mesh_median_ms = robust_median(latest_mesh_ms) or 0.0

    # offenders bookkeeping
    offenders = {
        "worst_node_correction": None,
        "stalest_node": None,
        "worst_link_sigma": None,
        "most_slew_clipped": None,
    }
    worst_corr = -1.0
    worst_age = -1.0
    worst_sigma = -1.0
    worst_clip = -1.0

    conv_cut = now - float(thr.conv_window_s)

    nodes_out: List[Dict[str, Any]] = []
    for nid in sorted(latest.keys()):
        r = latest[nid]

        created_at = _f(row_get(r, "created_at"))
        age_s = (now - created_at) if created_at is not None else None

        mesh_ms_abs = _proxy_mesh_time_ms(r)
        mesh_ms_centered = (mesh_ms_abs - mesh_median_ms) if mesh_ms_abs is not None else None

        offset_ms = _offset_ms(r)

        # recent controller stats in conv window
        recent = []
        for x in by_node_recent.get(nid, []):
            tx = _f(row_get(x, "created_at"))
            if tx is not None and tx >= conv_cut:
                recent.append(x)

        # correction magnitude
        deltas_applied = []
        clipped = []
        for x in recent:
            da = _f(row_get(x, "delta_applied_ms"))
            if da is not None:
                deltas_applied.append(abs(da))
            sc = _i(row_get(x, "slew_clipped"))
            if sc is not None:
                clipped.append(1 if sc else 0)

        med_abs_delta_applied_ms = robust_median(deltas_applied)
        clip_rate = (sum(clipped) / len(clipped)) if clipped else None

        # offset rate (ms/s) from recent node-only rows (median slope)
        slopes = []
        prev_t = None
        prev_off = None
        for x in recent:
            tx = _f(row_get(x, "created_at"))
            offx = _offset_ms(x)
            if tx is None or offx is None:
                continue
            if prev_t is not None and prev_off is not None:
                dt = tx - prev_t
                if dt > 1e-6:
                    slopes.append((offx - prev_off) / dt)
            prev_t, prev_off = tx, offx
        offset_rate_ms_s = robust_median(slopes)

        # link stability: consider outgoing links from node within conv window
        link_sigmas_recent = []
        link_samples_count = 0
        for lid, rows in by_link.items():
            if not lid.startswith(nid + "->"):
                continue
            for lr in rows:
                t_lr = _f(row_get(lr, "created_at"))
                if t_lr is None or t_lr < conv_cut:
                    continue
                s = _f(row_get(lr, "sigma_ms"))
                if s is not None:
                    link_sigmas_recent.append(s)
                    link_samples_count += 1

        link_sigma_med = robust_median(link_sigmas_recent)

        # deterministic convergence
        enough_samples = len(recent) >= thr.min_samples_warmup
        fresh_ok = (age_s is not None and age_s <= thr.fresh_s)
        correction_ok = (med_abs_delta_applied_ms is not None and med_abs_delta_applied_ms <= thr.delta_applied_med_ms)
        clip_ok = (clip_rate is None) or (clip_rate <= thr.slew_clip_rate)

        # link_ok rules:
        # - if we have enough node samples, we also want at least some link sigma evidence;
        # - then require median sigma below threshold.
        have_link_evidence = (link_samples_count >= thr.min_link_samples)
        link_ok = have_link_evidence and (link_sigma_med is not None) and (link_sigma_med <= thr.link_sigma_med_ms)

        if not enough_samples:
            state = "YELLOW"
            reason = "warming up / too few samples"
        elif not fresh_ok:
            state = "RED"
            reason = f"stale data (age {fmt(age_s,'.1f','n/a')}s)"
        else:
            # now data is fresh + enough samples
            if correction_ok and clip_ok and link_ok:
                state = "GREEN"
                reason = "converged"
            else:
                # if link evidence missing, that's an operator problem (can't assess stability)
                reasons = []
                if not correction_ok:
                    reasons.append(f"|Δapplied|med {fmt(med_abs_delta_applied_ms,'.2f','n/a')}ms")
                if not clip_ok and clip_rate is not None:
                    reasons.append(f"slew_clipped {fmt(clip_rate*100.0,'.0f','n/a')}%")
                if not have_link_evidence:
                    reasons.append("link samples missing")
                elif not link_ok:
                    reasons.append(f"linkσmed {fmt(link_sigma_med,'.2f','n/a')}ms")
                state = "YELLOW" if reasons else "YELLOW"
                reason = ", ".join(reasons) if reasons else "not converged"

        # offenders
        if med_abs_delta_applied_ms is not None and med_abs_delta_applied_ms > worst_corr:
            worst_corr = med_abs_delta_applied_ms
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

            "last_seen_created_at": created_at,
            "last_seen_utc": _utc(created_at),
            "age_s": age_s,

            "mesh_ms_abs": mesh_ms_abs,
            "mesh_ms_centered": mesh_ms_centered,
            "mesh_median_ms": mesh_median_ms,

            "offset_ms": offset_ms,
            "offset_rate_ms_s": offset_rate_ms_s,

            "med_abs_delta_applied_ms": med_abs_delta_applied_ms,
            "slew_clip_rate": clip_rate,

            "link_sigma_med_ms": link_sigma_med,
            "link_sigma_samples": link_samples_count,
        })

    # worst link sigma (median over conv window per link, choose max)
    for lid, rows in by_link.items():
        sigs = []
        for lr in rows:
            t_lr = _f(row_get(lr, "created_at"))
            if t_lr is None or t_lr < conv_cut:
                continue
            s = _f(row_get(lr, "sigma_ms"))
            if s is not None:
                sigs.append(s)
        med = robust_median(sigs)
        if med is not None and med > worst_sigma:
            worst_sigma = med
            offenders["worst_link_sigma"] = lid

    return {
        "nodes": nodes_out,
        "offenders": offenders,
        "meta": {
            "now": now,
            "now_utc": _utc(now),
            "window_s": eff_window,
            "thresholds": {
                "fresh_s": thr.fresh_s,
                "delta_applied_med_ms": thr.delta_applied_med_ms,
                "slew_clip_rate": thr.slew_clip_rate,
                "link_sigma_med_ms": thr.link_sigma_med_ms,
                "min_samples_warmup": thr.min_samples_warmup,
                "min_link_samples": thr.min_link_samples,
                "conv_window_s": thr.conv_window_s,
            },
        },
    }


def get_nodes_timeseries(window_s: float, bin_s: float, max_rows: int) -> Dict[str, Any]:
    """
    Returns:
      series[node] = [{t, mesh_ms_abs, mesh_ms_centered, offset_ms, offset_centered, offset_rate_ms_s}]
    """
    window_s = float(max(5.0, window_s))
    bin_s = float(max(0.1, bin_s))

    rows = fetch_node_rows(window_s=window_s, limit=int(max_rows))
    if not rows:
        return {"series": {}, "meta": {"x_axis": "created_at", "window_s": window_s, "bin_s": bin_s}}

    # collect times
    t_list = []
    for r in rows:
        t = _f(row_get(r, "created_at"))
        if t is not None:
            t_list.append(t)
    if not t_list:
        return {"series": {}, "meta": {"x_axis": "created_at", "window_s": window_s, "bin_s": bin_s}}

    t_min = min(t_list)

    # bins[idx][node] = (created_at, mesh_ms_abs, offset_ms)
    bins: Dict[int, Dict[str, Tuple[float, Optional[float], Optional[float]]]] = {}

    for r in rows:
        nid = str(row_get(r, "node_id", "") or "")
        t = _f(row_get(r, "created_at"))
        if not nid or t is None:
            continue
        mesh_ms = _proxy_mesh_time_ms(r)
        off_ms = _offset_ms(r)

        idx = int((t - t_min) / bin_s)
        if idx < 0:
            continue

        bucket = bins.setdefault(idx, {})
        prev = bucket.get(nid)
        if prev is None or t >= prev[0]:
            bucket[nid] = (t, mesh_ms, off_ms)

    # centered by per-bin mesh median (or offset median if mesh missing)
    # Build per bin: list of mesh candidates; fallback to offset.
    centered_bins: Dict[int, Dict[str, Dict[str, float]]] = {}

    for idx, bucket in bins.items():
        mesh_vals = []
        off_vals = []
        for _nid, (_t, mesh_ms, off_ms) in bucket.items():
            if mesh_ms is not None:
                mesh_vals.append(mesh_ms)
            if off_ms is not None:
                off_vals.append(off_ms)

        mesh_center = robust_median(mesh_vals)
        off_center = robust_median(off_vals)

        # if mesh_center missing, still compute offset-centered
        for nid, (_t, mesh_ms, off_ms) in bucket.items():
            o: Dict[str, float] = {}
            if mesh_ms is not None:
                o["mesh_ms_abs"] = float(mesh_ms)
                if mesh_center is not None:
                    o["mesh_ms_centered"] = float(mesh_ms - mesh_center)
            if off_ms is not None:
                o["offset_ms"] = float(off_ms)
                if off_center is not None:
                    o["offset_centered_ms"] = float(off_ms - off_center)
            centered_bins.setdefault(idx, {})[nid] = o

    # assemble series
    series: Dict[str, List[Dict[str, Any]]] = {}
    for idx in sorted(centered_bins.keys()):
        t_bin = t_min + (idx + 0.5) * bin_s
        bucket = centered_bins[idx]
        for nid, o in bucket.items():
            pt = {"t": float(t_bin)}
            pt.update(o)
            series.setdefault(nid, []).append(pt)

    # offset rate per node (local median slope over the series)
    for nid, pts in series.items():
        pts.sort(key=lambda p: p["t"])
        slopes = []
        prev = None
        for p in pts:
            if "offset_ms" not in p:
                prev = p
                continue
            if prev is not None and "offset_ms" in prev:
                dt = float(p["t"]) - float(prev["t"])
                if dt > 1e-6:
                    slopes.append((float(p["offset_ms"]) - float(prev["offset_ms"])) / dt)
            prev = p
        rate = robust_median(slopes)
        if rate is not None:
            for p in pts:
                p["offset_rate_ms_s"] = float(rate)

    return {"series": series, "meta": {"x_axis": "created_at", "window_s": window_s, "bin_s": bin_s}}


def get_links_table(window_s: float, conv_window_s: float, max_rows: int, fresh_s: float) -> Dict[str, Any]:
    """
    Returns per directed link:
      { link_id, node_id, peer_id,
        theta_last, theta_med,
        rtt_last,   rtt_med,
        sigma_last, sigma_med,
        last_seen_created_at, age_s,
        n, state }
    """
    now = time.time()
    window_s = float(max(5.0, window_s))
    conv_cut = now - float(conv_window_s)

    rows = fetch_link_rows(window_s=window_s, limit=int(max_rows))
    if not rows:
        return {"links": [], "meta": {"x_axis": "created_at", "window_s": window_s}}

    by_link: Dict[str, List[sqlite3.Row]] = {}
    for r in rows:
        nid = str(row_get(r, "node_id", "") or "")
        pid = str(row_get(r, "peer_id", "") or "")
        if not nid or not pid:
            continue
        lid = f"{nid}->{pid}"
        by_link.setdefault(lid, []).append(r)

    out: List[Dict[str, Any]] = []
    for lid in sorted(by_link.keys()):
        rs = by_link[lid]
        # last = most recent created_at
        rs_sorted = sorted(rs, key=lambda r: (_f(row_get(r, "created_at")) or 0.0))
        last = rs_sorted[-1] if rs_sorted else None

        t_last = _f(row_get(last, "created_at")) if last is not None else None
        age_s = (now - t_last) if t_last is not None else None

        theta_vals = []
        rtt_vals = []
        sigma_vals = []
        sigma_vals_conv = []
        for r in rs_sorted:
            th = _f(row_get(r, "theta_ms"))
            rt = _f(row_get(r, "rtt_ms"))
            sg = _f(row_get(r, "sigma_ms"))
            if th is not None:
                theta_vals.append(th)
            if rt is not None:
                rtt_vals.append(rt)
            if sg is not None:
                sigma_vals.append(sg)
                t = _f(row_get(r, "created_at"))
                if t is not None and t >= conv_cut:
                    sigma_vals_conv.append(sg)

        theta_med = robust_median(theta_vals)
        rtt_med = robust_median(rtt_vals)
        sigma_med = robust_median(sigma_vals)

        # state
        if age_s is None:
            state = "DOWN"
        else:
            state = "UP" if age_s <= float(fresh_s) else "DOWN"

        # stability (for operator use) is based on conv window sigma median
        sigma_med_conv = robust_median(sigma_vals_conv)
        stable = (sigma_med_conv is not None) and (sigma_med_conv <= THRESH_LINK_SIGMA_MED_MS) and (len(sigma_vals_conv) >= MIN_LINK_SAMPLES_FOR_STABLE)

        nid, pid = lid.split("->", 1)

        out.append({
            "link_id": lid,
            "node_id": nid,
            "peer_id": pid,

            "theta_last": _f(row_get(last, "theta_ms")) if last is not None else None,
            "theta_med": theta_med,

            "rtt_last": _f(row_get(last, "rtt_ms")) if last is not None else None,
            "rtt_med": rtt_med,

            "sigma_last": _f(row_get(last, "sigma_ms")) if last is not None else None,
            "sigma_med": sigma_med,

            "sigma_med_conv": sigma_med_conv,
            "stable_conv": bool(stable),

            "last_seen_created_at": t_last,
            "last_seen_utc": _utc(t_last),
            "age_s": age_s,

            "n": int(len(rs_sorted)),
            "state": state,
        })

    return {
        "links": out,
        "meta": {
            "x_axis": "created_at",
            "window_s": window_s,
            "fresh_s": fresh_s,
            "conv_window_s": conv_window_s,
        },
    }


def get_heatmap(window_s: float, bin_s: float, max_rows: int, metric: str) -> Dict[str, Any]:
    """
    Heatmap shows exactly one metric per mode:
      metric=theta -> theta_ms (abs optional in UI)
      metric=sigma -> sigma_ms
      metric=freshness -> age_s (computed per-bin from last sample time)
    Rows are directed links (node->peer), columns are time bins.
    """
    metric = str(metric or "sigma").lower().strip()
    if metric not in ("theta", "sigma", "freshness"):
        metric = "sigma"

    window_s = float(max(5.0, window_s))
    bin_s = float(max(0.1, bin_s))

    rows = fetch_link_rows(window_s=window_s, limit=int(max_rows))
    if not rows:
        return {"data": [], "links": [], "n_bins": 0, "metric": metric, "meta": {"x_axis": "created_at"}}

    # collect times
    t_list = []
    for r in rows:
        t = _f(row_get(r, "created_at"))
        if t is not None:
            t_list.append(t)
    if not t_list:
        return {"data": [], "links": [], "n_bins": 0, "metric": metric, "meta": {"x_axis": "created_at"}}

    t_min = min(t_list)
    t_max = max(t_list)
    total_bins = max(1, int((t_max - t_min) / bin_s) + 1)

    # downsample bins for UI stability
    display_bins = min(HEATMAP_MAX_BINS, total_bins)
    bin_scale = total_bins / display_bins

    # bins[d_idx][link] = (t_last, value or None)
    bins: Dict[int, Dict[str, Tuple[float, Optional[float]]]] = {}

    for r in rows:
        t = _f(row_get(r, "created_at"))
        nid = str(row_get(r, "node_id", "") or "")
        pid = str(row_get(r, "peer_id", "") or "")
        if t is None or not nid or not pid:
            continue
        lid = f"{nid}->{pid}"

        # map raw bin index to display bin index
        bi = int((t - t_min) / bin_s)
        rel = bi / max(1.0, float(total_bins))
        d_idx = int(rel * display_bins)
        if d_idx >= display_bins:
            d_idx = display_bins - 1
        if d_idx < 0:
            continue

        # value
        v = None
        if metric == "theta":
            v = _f(row_get(r, "theta_ms"))
        elif metric == "sigma":
            v = _f(row_get(r, "sigma_ms"))
        elif metric == "freshness":
            # placeholder: computed after we pick last per bin (we store t)
            v = 0.0

        bucket = bins.setdefault(d_idx, {})
        prev = bucket.get(lid)
        if prev is None or t >= prev[0]:
            bucket[lid] = (t, v)

    # compute freshness ages if needed
    now = time.time()
    if metric == "freshness":
        for d_idx, bucket in bins.items():
            for lid, (t_last, _v) in list(bucket.items()):
                age = now - t_last
                bucket[lid] = (t_last, float(age))

    # flatten for chart.js matrix
    link_ids = sorted({lid for bucket in bins.values() for lid in bucket.keys()})
    idx_map = {lid: i for i, lid in enumerate(link_ids)}

    data = []
    for d_idx in range(display_bins):
        bucket = bins.get(d_idx, {})
        # center time for this display bin
        idx_center = (d_idx + 0.5) * bin_scale
        t_center = t_min + idx_center * bin_s
        for lid, (_t_last, v) in bucket.items():
            if v is None:
                continue
            data.append({"x": float(t_center), "y": int(idx_map.get(lid, 0)), "v": float(v)})

    return {
        "data": data,
        "links": link_ids,
        "n_bins": int(display_bins),
        "metric": metric,
        "meta": {"x_axis": "created_at", "window_s": window_s, "bin_s": bin_s},
    }


def get_controller_timeseries(window_s: float, max_rows: int, node: Optional[str]) -> Dict[str, Any]:
    """
    Debug-only: return timeseries for a single node (or all if node=None, but UI should call per-node).
    """
    window_s = float(max(5.0, window_s))
    rows = fetch_node_rows(window_s=window_s, limit=int(max_rows))
    if not rows:
        return {"node": node, "series": [], "meta": {"x_axis": "created_at", "window_s": window_s}}

    node = (str(node).strip() if node is not None else None)
    out = []
    for r in rows:
        nid = str(row_get(r, "node_id", "") or "")
        if node and nid != node:
            continue
        t = _f(row_get(r, "created_at"))
        if t is None:
            continue

        pt: Dict[str, Any] = {"t": float(t)}
        for k in ("delta_desired_ms", "delta_applied_ms", "dt_s", "slew_clipped"):
            if not row_has(r, k):
                continue
            if k == "slew_clipped":
                v = _i(row_get(r, k))
                if v is not None:
                    pt[k] = 1 if v else 0
            else:
                v = _f(row_get(r, k))
                if v is not None:
                    pt[k] = float(v)
        out.append(pt)

    out.sort(key=lambda p: p["t"])
    return {"node": node, "series": out, "meta": {"x_axis": "created_at", "window_s": window_s}}


# -----------------------------
# Routes (final API)
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
        sync_cfg = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync_cfg.get("ui_window_s", WINDOW_S_DEFAULT))
        return jsonify(compute_overview(window_s=window_s))
    except Exception as e:
        return _json_error(f"/api/overview failed: {e}", 500)


@app.route("/api/nodes_timeseries")
def api_nodes_timeseries():
    try:
        cfg = load_config()
        sync_cfg = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(request.args.get("window", sync_cfg.get("ui_window_s", WINDOW_S_DEFAULT)))
        bin_s = float(request.args.get("bin_s", sync_cfg.get("ui_bin_s", BIN_S_DEFAULT)))
        max_rows = int(request.args.get("max_rows", sync_cfg.get("ui_max_points", NODE_MAX_ROWS_DEFAULT)))
        return jsonify(get_nodes_timeseries(window_s=window_s, bin_s=bin_s, max_rows=max_rows))
    except Exception as e:
        return _json_error(f"/api/nodes_timeseries failed: {e}", 500)


@app.route("/api/links_table")
def api_links_table():
    try:
        cfg = load_config()
        sync_cfg = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(request.args.get("window", sync_cfg.get("ui_window_s", WINDOW_S_DEFAULT)))
        max_rows = int(request.args.get("max_rows", sync_cfg.get("ui_link_max_points", LINK_MAX_ROWS_DEFAULT)))
        fresh_s = _fresh_threshold_s_from_cfg(cfg)
        return jsonify(get_links_table(window_s=window_s, conv_window_s=CONV_WINDOW_S, max_rows=max_rows, fresh_s=fresh_s))
    except Exception as e:
        return _json_error(f"/api/links_table failed: {e}", 500)


@app.route("/api/heatmap")
def api_heatmap():
    try:
        cfg = load_config()
        sync_cfg = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(request.args.get("window", sync_cfg.get("ui_window_s", WINDOW_S_DEFAULT)))
        bin_s = float(request.args.get("bin_s", sync_cfg.get("ui_bin_s", BIN_S_DEFAULT)))
        metric = str(request.args.get("metric", "sigma"))
        max_rows = int(request.args.get("max_rows", sync_cfg.get("ui_link_max_points", LINK_MAX_ROWS_DEFAULT)))
        return jsonify(get_heatmap(window_s=window_s, bin_s=bin_s, max_rows=max_rows, metric=metric))
    except Exception as e:
        return _json_error(f"/api/heatmap failed: {e}", 500)


@app.route("/api/controller")
def api_controller():
    try:
        cfg = load_config()
        sync_cfg = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(request.args.get("window", sync_cfg.get("ui_window_s", WINDOW_S_DEFAULT)))
        max_rows = int(request.args.get("max_rows", sync_cfg.get("ui_ctrl_max_points", NODE_MAX_ROWS_DEFAULT)))
        node = request.args.get("node", None)
        return jsonify(get_controller_timeseries(window_s=window_s, max_rows=max_rows, node=node))
    except Exception as e:
        return _json_error(f"/api/controller failed: {e}", 500)


@app.template_filter("datetime_utc")
def datetime_utc(ts):
    return _utc(_f(ts))


# -----------------------------
# UI (Operator vs Debug, Tabs)
# -----------------------------
TEMPLATE = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>MeshTime Dashboard</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">

  <style>
    :root{
      --bg:#0f0f10;
      --card:#18181a;
      --card2:#141416;
      --text:#eaeaea;
      --muted:rgba(255,255,255,0.68);
      --line:rgba(255,255,255,0.08);
      --ok:#2ecc71;
      --warn:#f1c40f;
      --bad:#e74c3c;
      --accent:#52a7ff;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    body { margin:0; background:var(--bg); color:var(--text); font-family:var(--sans); }
    .wrap{ padding: 1.25rem; max-width: 1480px; margin: 0 auto; }

    .topbar{ display:flex; gap:1rem; align-items:center; justify-content:space-between; flex-wrap:wrap; }
    h1{ margin:0; font-size:1.25rem; letter-spacing:0.2px; }
    .sub{ margin-top:0.25rem; font-size:0.85rem; color:var(--muted); }
    .mono{ font-family:var(--mono); }
    .muted{ color:var(--muted); }

    .card{ background:var(--card); border: 1px solid var(--line); border-radius:14px; padding: 1rem; }
    .card2{ background:var(--card2); border: 1px solid var(--line); border-radius:12px; padding: 0.85rem; }

    .row{ display:flex; gap:0.75rem; align-items:center; flex-wrap:wrap; }
    .spacer{ height:0.8rem; }

    .toggle{ display:flex; gap:0.6rem; align-items:center; padding:0.55rem 0.8rem; }
    .toggle input{ transform: scale(1.15); }

    .tabs{ display:flex; gap:0.5rem; flex-wrap:wrap; }
    .tabbtn{
      cursor:pointer; user-select:none;
      padding: 0.5rem 0.75rem; border-radius: 999px;
      border: 1px solid var(--line); background: rgba(255,255,255,0.02);
      color: var(--text); font-size:0.9rem;
    }
    .tabbtn.active{ border-color: rgba(82,167,255,0.55); box-shadow: 0 0 0 1px rgba(82,167,255,0.25) inset; }

    .grid2{ display:grid; grid-template-columns: minmax(0, 1.4fr) minmax(0, 1fr); gap:1rem; align-items:start; }
    @media (max-width: 1150px){ .grid2{ grid-template-columns: minmax(0,1fr); } }

    .kpiGrid{ display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 0.75rem; }
    @media (max-width: 1150px){ .kpiGrid{ grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 750px){ .kpiGrid{ grid-template-columns: minmax(0, 1fr); } }

    .kpiTitle{ font-size:0.85rem; color:var(--muted); }
    .kpiValue{ font-size:1.35rem; font-weight: 780; margin-top:0.25rem; letter-spacing:0.1px; }
    .kpiHint{ font-size:0.82rem; color:var(--muted); margin-top:0.2rem; }
    .kpiReason{ font-size:0.82rem; margin-top:0.35rem; color:var(--muted); }

    .pill{ display:inline-block; padding: 0.12rem 0.55rem; border-radius:999px; font-size:0.75rem; font-weight: 750; }
    .pill.ok{ background: rgba(46,204,113,0.14); color: var(--ok); }
    .pill.warn{ background: rgba(241,196,15,0.14); color: var(--warn); }
    .pill.bad{ background: rgba(231,76,60,0.14); color: var(--bad); }

    table { width:100%; border-collapse: collapse; font-size:0.9rem; }
    th, td { padding: 0.45rem 0.5rem; text-align:left; vertical-align:top; }
    th { border-bottom: 1px solid rgba(255,255,255,0.14); font-weight: 750; color: var(--muted); }
    tr:nth-child(even) td { background: rgba(255,255,255,0.02); }

    .page{ display:none; }
    .page.active{ display:block; }

    canvas{ width:100%; max-width:100%; }

    #meshCanvas{ height: 280px; background: #101012; border-radius: 12px; border:1px solid var(--line); }

    .chartGrid{ display:grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 1rem; }
    @media (max-width: 1100px){ .chartGrid{ grid-template-columns: minmax(0,1fr); } }

    .debugOnly{ display:none; }
    .debugOn .debugOnly{ display:block; }

    .rightColSticky{
      position: sticky;
      top: 1rem;
    }
    @media (max-width: 1150px){ .rightColSticky{ position: static; } }

    select, button{
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--line);
      color: var(--text);
      border-radius: 10px;
      padding: 0.45rem 0.55rem;
      font-size: 0.9rem;
    }
    .small{ font-size:0.85rem; }
  </style>

  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-matrix@1.3.0/dist/chartjs-chart-matrix.min.js"></script>
</head>

<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <h1>MeshTime Dashboard</h1>
        <div class="sub">
          Datenquelle: <span class="mono">{{ db_path }}</span> · X-Achse: <b>created_at (Sink-Clock)</b>
        </div>
      </div>

      <div class="row">
        <div class="card toggle">
          <input id="debugToggle" type="checkbox">
          <label for="debugToggle" class="small">Debug-Mode</label>
        </div>

        <div class="tabs">
          <div class="tabbtn active" data-tab="overview">Overview</div>
          <div class="tabbtn" data-tab="meshtime">Mesh-Time</div>
          <div class="tabbtn" data-tab="links">Links</div>
          <div class="tabbtn" data-tab="heatmap">Heatmap</div>
          <div class="tabbtn debugOnly" data-tab="controller">Controller</div>
        </div>
      </div>
    </div>

    <div class="spacer"></div>

    <!-- Overview -->
    <div class="page active" id="page-overview">
      <div class="grid2">
        <div class="card">
          <div class="row" style="justify-content:space-between;">
            <div>
              <div style="font-size:1.05rem; font-weight:800;">Ist das Mesh synchronisiert?</div>
              <div class="muted small" id="ovMeta">lade…</div>
            </div>
            <div class="row">
              <span class="pill ok" id="meshAmpel" style="display:none;">🟢</span>
              <span class="pill warn" id="meshAmpel2" style="display:none;">🟡</span>
              <span class="pill bad" id="meshAmpel3" style="display:none;">🔴</span>
              <span class="muted small" id="meshAmpelText">—</span>
            </div>
          </div>

          <div class="spacer"></div>

          <div class="kpiGrid" id="nodeKpis">
            <div class="card2">
              <div class="kpiTitle">Nodes</div>
              <div class="kpiValue">…</div>
              <div class="kpiHint">—</div>
            </div>
          </div>

          <div class="spacer"></div>

          <div class="row" style="justify-content:space-between; align-items:flex-end;">
            <div>
              <div style="font-size:1rem; font-weight:800;">Worst Offenders</div>
              <div class="muted small">Wenn’s knirscht: hier anfangen.</div>
            </div>
          </div>

          <div class="spacer" style="height:0.4rem;"></div>
          <div class="small muted" id="offendersLine">lade…</div>
        </div>

        <div class="rightColSticky">
          <div class="card">
            <div class="row" style="justify-content:space-between;">
              <div>
                <div style="font-size:1rem; font-weight:800;">Topologie</div>
                <div class="muted small">Nodes: Ampel · Links: Qualität (σ)</div>
              </div>
              <div class="row">
                <span class="pill ok">σ gut</span>
                <span class="pill warn">σ mittel</span>
                <span class="pill bad">σ schlecht</span>
              </div>
            </div>
            <div class="spacer" style="height:0.6rem;"></div>
            <canvas id="meshCanvas"></canvas>
            <div class="muted small" style="margin-top:0.6rem;">
              Operator-Logik: Mesh-Time ist pro Node sichtbar. Deltas/Links erklären “warum”.
            </div>
          </div>

          <div class="spacer"></div>

          <div class="card">
            <div style="font-size:1rem; font-weight:800;">Links (Scan-Table)</div>
            <div class="muted small">θ (Offset), RTT, σ (Stability), UP/DOWN, Age</div>
            <div class="spacer" style="height:0.6rem;"></div>

            <div style="overflow:auto; max-height: 310px;">
              <table id="linksMiniTable">
                <thead>
                  <tr>
                    <th>Link</th>
                    <th>State</th>
                    <th>σ med</th>
                    <th>RTT med</th>
                    <th>Age</th>
                  </tr>
                </thead>
                <tbody><tr><td colspan="5" class="muted small">lade…</td></tr></tbody>
              </table>
            </div>

          </div>
        </div>
      </div>
    </div>

    <!-- Mesh-Time -->
    <div class="page" id="page-meshtime">
      <div class="card">
        <div class="row" style="justify-content:space-between;">
          <div>
            <div style="font-size:1.05rem; font-weight:800;">Uhrenansicht</div>
            <div class="muted small">Primär: centered (t_mesh - mesh_median). Debug: absolute.</div>
          </div>
          <div class="row">
            <label class="muted small">View:</label>
            <select id="meshViewSel">
              <option value="centered">Centered (empfohlen)</option>
              <option value="absolute">Absolute (Debug)</option>
            </select>
          </div>
        </div>

        <div class="spacer"></div>

        <div class="card2">
          <div class="muted small">t_mesh(node) – Konsens über Zeit</div>
          <canvas id="meshTimeChart" height="140"></canvas>
        </div>

        <div class="spacer"></div>

        <div class="row" style="justify-content:space-between;">
          <div>
            <div style="font-size:1rem; font-weight:800;">Node Tabelle</div>
            <div class="muted small">current mesh-time, offset, offset-rate, converged, last_seen</div>
          </div>
        </div>
        <div class="spacer" style="height:0.5rem;"></div>

        <div style="overflow:auto;">
          <table id="nodesTable">
            <thead>
              <tr>
                <th>Node</th>
                <th>Status</th>
                <th>Mesh-Time (centered)</th>
                <th>Offset</th>
                <th>Offset-Rate</th>
                <th>Age</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody><tr><td colspan="7" class="muted small">lade…</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- Links -->
    <div class="page" id="page-links">
      <div class="card">
        <div class="row" style="justify-content:space-between;">
          <div>
            <div style="font-size:1.05rem; font-weight:800;">Link Metrics</div>
            <div class="muted small">Diagnose: warum nicht synchron? (θ, RTT, σ, Freshness)</div>
          </div>
        </div>

        <div class="spacer"></div>

        <div style="overflow:auto;">
          <table id="linksTable">
            <thead>
              <tr>
                <th>Link</th>
                <th>State</th>
                <th>θ last</th>
                <th>θ med</th>
                <th>RTT last</th>
                <th>RTT med</th>
                <th>σ last</th>
                <th>σ med</th>
                <th>Age</th>
                <th>n</th>
                <th>Stable (conv)</th>
              </tr>
            </thead>
            <tbody><tr><td colspan="11" class="muted small">lade…</td></tr></tbody>
          </table>
        </div>

        <div class="spacer"></div>

        <div class="chartGrid debugOnly">
          <div class="card2">
            <div class="muted small">θ pro Link (ms)</div>
            <canvas id="thetaChart" height="140"></canvas>
          </div>
          <div class="card2">
            <div class="muted small">σ pro Link (ms)</div>
            <canvas id="sigmaChart" height="140"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- Heatmap -->
    <div class="page" id="page-heatmap">
      <div class="card">
        <div class="row" style="justify-content:space-between;">
          <div>
            <div style="font-size:1.05rem; font-weight:800;">Heatmap</div>
            <div class="muted small">Genau 1 Metrik pro Modus. Stabil gerendert.</div>
          </div>
          <div class="row">
            <label class="muted small">Metric:</label>
            <select id="heatMetricSel">
              <option value="sigma">σ (Stability)</option>
              <option value="theta">θ (Offset)</option>
              <option value="freshness">Freshness (age_s)</option>
            </select>
          </div>
        </div>

        <div class="spacer"></div>

        <div class="card2">
          <canvas id="heatmapChart" height="160"></canvas>
          <div class="muted small" style="margin-top:0.5rem;" id="heatMeta">—</div>
        </div>
      </div>
    </div>

    <!-- Controller (Debug) -->
    <div class="page debugOnly" id="page-controller">
      <div class="card">
        <div class="row" style="justify-content:space-between;">
          <div>
            <div style="font-size:1.05rem; font-weight:800;">Controller (Debug)</div>
            <div class="muted small">delta_desired/applied, dt, slew_clipped → Ursache/Wirkung</div>
          </div>
          <div class="row">
            <label class="muted small">Node:</label>
            <select id="ctrlNodeSel"></select>
          </div>
        </div>

        <div class="spacer"></div>

        <div class="chartGrid">
          <div class="card2">
            <div class="muted small">delta_desired_ms</div>
            <canvas id="deltaDesiredChart" height="140"></canvas>
          </div>
          <div class="card2">
            <div class="muted small">delta_applied_ms</div>
            <canvas id="deltaAppliedChart" height="140"></canvas>
          </div>
          <div class="card2">
            <div class="muted small">dt_s</div>
            <canvas id="dtChart" height="140"></canvas>
          </div>
          <div class="card2">
            <div class="muted small">slew_clipped (0/1)</div>
            <canvas id="slewChart" height="140"></canvas>
          </div>
        </div>
      </div>
    </div>

  </div>

<script>
  const COLORS = [
    'rgba(46, 204, 113, 0.55)',
    'rgba(82, 167, 255, 0.55)',
    'rgba(241, 196, 15, 0.55)',
    'rgba(231, 76, 60, 0.55)',
    'rgba(155, 89, 182, 0.55)',
    'rgba(26, 188, 156, 0.55)',
    'rgba(230, 126, 34, 0.55)',
  ];

  function stroke(c){ return c.replace('0.55', '0.95'); }
  function pillClass(state){
    if(state === 'GREEN') return 'pill ok';
    if(state === 'RED') return 'pill bad';
    return 'pill warn';
  }
  function pillText(state){
    if(state === 'GREEN') return 'GREEN';
    if(state === 'RED') return 'RED';
    return 'YELLOW';
  }

  async function fetchJson(url){
    const resp = await fetch(url);
    let data = null;
    try { data = await resp.json(); } catch(e) { data = {error:'invalid json'}; }
    if(!resp.ok) throw new Error(data.error || ('HTTP ' + resp.status));
    return data;
  }

  // -------- Tabs / Debug mode ----------
  function setTab(tab){
    document.querySelectorAll('.tabbtn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const el = document.getElementById('page-' + tab);
    if(el) el.classList.add('active');
  }

  function applyDebug(){
    const on = document.getElementById('debugToggle').checked;
    document.body.classList.toggle('debugOn', on);
    // if controller tab active but debug turned off -> go back to overview
    const active = document.querySelector('.tabbtn.active')?.dataset?.tab || 'overview';
    if(!on && active === 'controller') setTab('overview');
  }

  // -------- Charts ----------
  function lineChart(ctx, ySuffix){
    return new Chart(ctx, {
      type:'line',
      data:{ datasets:[] },
      options:{
        responsive:true,
        animation:false,
        scales:{
          x:{ type:'time', time:{ unit:'second' }, ticks:{ color:'#aaa' }, grid:{ color:'rgba(255,255,255,0.06)' } },
          y:{ ticks:{ color:'#aaa', callback:(v)=> (v?.toFixed? v.toFixed(2):v) + (ySuffix||'') }, grid:{ color:'rgba(255,255,255,0.06)' } }
        },
        plugins:{ legend:{ labels:{ color:'#eee' } } },
        elements:{ point:{ radius:0 }, line:{ tension:0.12 } }
      }
    });
  }

  function heatmapFactory(ctx){
    return new Chart(ctx, {
      type:'matrix',
      data:{ datasets:[{
        label:'Heatmap',
        data:[],
        borderWidth:0,
        backgroundColor:(context)=>{
          const raw = context.raw;
          if(!raw || typeof raw.v !== 'number') return 'rgba(0,0,0,0)';
          const maxV = context.chart._maxV || 1;
          const ratio = Math.min(1, Math.abs(raw.v)/maxV);
          const r = Math.round(255 * ratio);
          const g = Math.round(255 * (1 - ratio));
          return `rgba(${r},${g},160,0.85)`;
        },
        width:(ctx)=>{
          const area = ctx.chart.chartArea || {};
          const n = ctx.chart._nBins || 1;
          const w = (area.right - area.left) / n;
          return (Number.isFinite(w) && w>0) ? w : 8;
        },
        height:(ctx)=>{
          const area = ctx.chart.chartArea || {};
          const cats = ctx.chart._cats || ['link'];
          const h = (area.bottom - area.top) / Math.max(1, cats.length);
          return (Number.isFinite(h) && h>0) ? h : 10;
        }
      }]},
      options:{
        responsive:true,
        animation:false,
        scales:{
          x:{ type:'time', time:{ unit:'second' }, ticks:{ color:'#aaa' }, grid:{ color:'rgba(255,255,255,0.06)' } },
          y:{
            type:'linear',
            ticks:{
              color:'#aaa',
              callback:function(v){
                const cats = this.chart._cats || [];
                const idx = Math.round(v);
                return cats[idx] || '';
              }
            },
            grid:{ color:'rgba(255,255,255,0.06)' }
          }
        },
        plugins:{ legend:{ labels:{ color:'#eee' } } }
      }
    });
  }

  let meshTimeChart, thetaChart, sigmaChart, heatChart;
  let deltaDesiredChart, deltaAppliedChart, dtChart, slewChart;

  function initCharts(){
    meshTimeChart = lineChart(document.getElementById('meshTimeChart').getContext('2d'), ' ms');
    thetaChart = lineChart(document.getElementById('thetaChart').getContext('2d'), ' ms');
    sigmaChart = lineChart(document.getElementById('sigmaChart').getContext('2d'), ' ms');

    heatChart = heatmapFactory(document.getElementById('heatmapChart').getContext('2d'));

    deltaDesiredChart = lineChart(document.getElementById('deltaDesiredChart').getContext('2d'), ' ms');
    deltaAppliedChart = lineChart(document.getElementById('deltaAppliedChart').getContext('2d'), ' ms');
    dtChart = lineChart(document.getElementById('dtChart').getContext('2d'), ' s');
    slewChart = lineChart(document.getElementById('slewChart').getContext('2d'), '');
  }

  function updateLineBySeries(chart, seriesMap, field, labelPrefix=''){
    const ids = Object.keys(seriesMap||{}).sort();
    chart.data.datasets = [];
    ids.forEach((id, idx)=>{
      const c = COLORS[idx % COLORS.length];
      const pts = (seriesMap[id]||[])
        .filter(p => p && p[field] !== null && p[field] !== undefined && Number.isFinite(p[field]))
        .map(p => ({ x: new Date(p.t*1000), y: p[field] }));
      chart.data.datasets.push({
        label: labelPrefix + id,
        data: pts,
        borderColor: stroke(c),
        backgroundColor: c,
        fill:false,
        borderWidth: 1.6,
        pointRadius: 0
      });
    });
    chart.update();
  }

  function updateControllerCharts(ctrlSeries){
    const s = (ctrlSeries || []).map(p => ({ t:p.t, dd:p.delta_desired_ms, da:p.delta_applied_ms, dt:p.dt_s, sc:p.slew_clipped }));
    const byNode = { "": s }; // reuse helper: fake id
    updateLineBySeries(deltaDesiredChart, byNode, "dd", "");
    updateLineBySeries(deltaAppliedChart, byNode, "da", "");
    updateLineBySeries(dtChart, byNode, "dt", "");
    updateLineBySeries(slewChart, byNode, "sc", "");
    // rename legend
    [deltaDesiredChart, deltaAppliedChart, dtChart, slewChart].forEach(ch => {
      if(ch.data.datasets[0]) ch.data.datasets[0].label = "selected node";
      ch.update();
    });
  }

  function updateHeatmap(hm){
    const data = (hm && hm.data) || [];
    const cats = (hm && hm.links) || [];
    let maxV = 1;
    data.forEach(d => { if(typeof d.v === 'number') maxV = Math.max(maxV, Math.abs(d.v)); });

    heatChart.data.datasets[0].data = data.map(d => ({
      x: new Date(d.x*1000),
      y: d.y,
      v: d.v
    }));
    heatChart._cats = cats;
    heatChart._nBins = hm.n_bins || 1;
    heatChart._maxV = maxV || 1;
    heatChart.update();

    const metric = hm.metric || 'sigma';
    const mtxt = metric === 'sigma' ? 'σ (ms)' : (metric === 'theta' ? 'θ (ms)' : 'freshness (age_s)');
    document.getElementById('heatMeta').textContent =
      `metric=${mtxt} · bins=${hm.n_bins || 0} · links=${cats.length}`;
  }

  // -------- Mesh drawing (nodes + link quality) ----------
  function linkColorBySigma(sig){
    if(sig === null || sig === undefined || !Number.isFinite(sig)) return {stroke:'rgba(255,255,255,0.22)', w:1.0};
    if(sig <= 2.0) return {stroke:'rgba(46,204,113,0.75)', w:2.2};
    if(sig <= 6.0) return {stroke:'rgba(241,196,15,0.75)', w:2.0};
    return {stroke:'rgba(231,76,60,0.75)', w:2.2};
  }

  function drawMesh(canvas, topo, nodeStates, linkSigmaMap){
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);

    const nodes = topo.nodes || [];
    const links = topo.links || [];
    if(!nodes.length){
      ctx.fillStyle = '#aaa';
      ctx.font = '12px system-ui';
      ctx.fillText('Keine Nodes in config/nodes.json gefunden.', 10, 20);
      return;
    }

    const n = nodes.length;
    const R = Math.min(w,h) * 0.36;
    const cx = w/2, cy = h/2;

    const pos = {};
    nodes.forEach((node, i)=>{
      const a = (2*Math.PI*i)/n - Math.PI/2;
      pos[node.id] = { x: cx + R*Math.cos(a), y: cy + R*Math.sin(a) };
    });

    // links (undirected display: pick directed sigma if exists; use worst of both directions)
    links.forEach(l=>{
      const a = pos[l.a], b = pos[l.b];
      if(!a || !b) return;

      const s1 = linkSigmaMap[l.a + '->' + l.b];
      const s2 = linkSigmaMap[l.b + '->' + l.a];
      let use = null;
      if(Number.isFinite(s1) && Number.isFinite(s2)) use = Math.max(s1, s2);
      else if(Number.isFinite(s1)) use = s1;
      else if(Number.isFinite(s2)) use = s2;

      const lc = linkColorBySigma(use);

      ctx.strokeStyle = lc.stroke;
      ctx.lineWidth = lc.w;
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    });

    nodes.forEach(node=>{
      const p = pos[node.id]; if(!p) return;
      const st = nodeStates[node.id] || 'YELLOW';
      const isRoot = !!node.is_root;
      const r = isRoot ? 20 : 16;

      let fill = 'rgba(241,196,15,0.16)';
      let stroke = 'rgba(241,196,15,0.95)';
      if(st === 'GREEN'){ fill='rgba(46,204,113,0.18)'; stroke='rgba(46,204,113,0.95)'; }
      if(st === 'RED'){ fill='rgba(231,76,60,0.18)'; stroke='rgba(231,76,60,0.95)'; }

      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,2*Math.PI);
      ctx.fillStyle = fill; ctx.fill();
      ctx.lineWidth = isRoot ? 3.2 : 2.2;
      ctx.strokeStyle = stroke; ctx.stroke();

      ctx.fillStyle = '#ecf0f1';
      ctx.font = '12px system-ui';
      ctx.textAlign='center'; ctx.textBaseline='middle';
      ctx.fillText(node.id, p.x, p.y);
    });

    const roots = nodes.filter(n=>n.is_root).map(n=>n.id);
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '11px system-ui';
    ctx.textAlign='left'; ctx.textBaseline='top';
    ctx.fillText(roots.length ? ('Root: ' + roots.join(', ')) : 'Root: keiner', 10, 10);
  }

  // -------- Renderers ----------
  function computeMeshAmpel(nodes){
    // Mesh-level: GREEN if all nodes green; RED if all red or no data; else YELLOW
    if(!nodes || !nodes.length) return {state:'RED', text:'no data'};
    const counts = {GREEN:0, YELLOW:0, RED:0};
    nodes.forEach(n => counts[n.state] = (counts[n.state]||0) + 1);

    if(counts.RED === nodes.length) return {state:'RED', text:'no fresh data / unstable'};
    if(counts.GREEN === nodes.length) return {state:'GREEN', text:'converged'};
    // if many are red -> red-ish
    if(counts.RED > 0 && counts.GREEN === 0) return {state:'RED', text:'not synchronized'};
    return {state:'YELLOW', text:'partial / warming up / mixed'};
  }

  function renderOverview(ov){
    const meta = ov.meta || {};
    const nodes = ov.nodes || [];
    const thr = meta.thresholds || {};
    const freshS = thr.fresh_s ?? '?';
    const dmed = thr.delta_applied_med_ms ?? '?';
    const clip = (thr.slew_clip_rate != null) ? Math.round(thr.slew_clip_rate*100) : '?';
    const lsig = thr.link_sigma_med_ms ?? '?';
    const minS = thr.min_samples_warmup ?? '?';
    const minL = thr.min_link_samples ?? '?';

    document.getElementById('ovMeta').textContent =
      `now=${meta.now_utc || 'n/a'} · conv_window=${thr.conv_window_s || '?'}s · thresholds: fresh≤${freshS}s, |Δapplied|med≤${dmed}ms, clip≤${clip}%, linkσmed≤${lsig}ms · warmup≥${minS} samples · link≥${minL} σ-samples`;

    // mesh-level ampel
    const am = computeMeshAmpel(nodes);
    document.getElementById('meshAmpel').style.display = (am.state==='GREEN') ? '' : 'none';
    document.getElementById('meshAmpel2').style.display = (am.state==='YELLOW') ? '' : 'none';
    document.getElementById('meshAmpel3').style.display = (am.state==='RED') ? '' : 'none';
    document.getElementById('meshAmpelText').textContent = am.text;

    // node cards
    const wrap = document.getElementById('nodeKpis');
    wrap.innerHTML = '';
    nodes.forEach((n, idx)=>{
      const centered = (n.mesh_ms_centered != null) ? (n.mesh_ms_centered.toFixed(2) + ' ms') : 'n/a';
      const age = (n.age_s != null) ? (n.age_s.toFixed(1) + ' s') : 'n/a';
      const off = (n.offset_ms != null) ? (n.offset_ms.toFixed(2) + ' ms') : '—';
      const rate = (n.offset_rate_ms_s != null) ? (n.offset_rate_ms_s.toFixed(3) + ' ms/s') : '—';
      const da = (n.med_abs_delta_applied_ms != null) ? (n.med_abs_delta_applied_ms.toFixed(2) + ' ms') : '—';
      const clipr = (n.slew_clip_rate != null) ? (Math.round(n.slew_clip_rate*100) + '%') : '—';
      const sig = (n.link_sigma_med_ms != null) ? (n.link_sigma_med_ms.toFixed(2) + ' ms') : '—';
      const ls = (n.link_sigma_samples != null) ? n.link_sigma_samples : 0;

      const div = document.createElement('div');
      div.className = 'card2';
      div.innerHTML = `
        <div class="row" style="justify-content:space-between;">
          <div class="kpiTitle">Node ${n.node_id}</div>
          <span class="${pillClass(n.state)}">${pillText(n.state)}</span>
        </div>
        <div class="kpiValue">${centered}</div>
        <div class="kpiHint">age ${age} · offset ${off} · rate ${rate}</div>
        <div class="kpiHint">|Δapplied|med ${da} · clip ${clipr} · linkσmed ${sig} (${ls})</div>
        <div class="kpiReason">${n.reason || ''}</div>
      `;
      wrap.appendChild(div);
    });

    const off = ov.offenders || {};
    document.getElementById('offendersLine').textContent =
      `worst correction: ${off.worst_node_correction || '—'} · stalest: ${off.stalest_node || '—'} · most clipped: ${off.most_slew_clipped || '—'} · worst link σ (conv med): ${off.worst_link_sigma || '—'}`;

    return nodes;
  }

  function renderNodesTable(nodes){
    const tb = document.querySelector('#nodesTable tbody');
    tb.innerHTML = '';
    if(!nodes || !nodes.length){
      tb.innerHTML = '<tr><td colspan="7" class="muted small">Keine Daten.</td></tr>';
      return;
    }
    nodes.forEach(n=>{
      const cls = pillClass(n.state);
      const centered = (n.mesh_ms_centered != null) ? `${n.mesh_ms_centered.toFixed(2)} ms` : 'n/a';
      const off = (n.offset_ms != null) ? `${n.offset_ms.toFixed(2)} ms` : '—';
      const rate = (n.offset_rate_ms_s != null) ? `${n.offset_rate_ms_s.toFixed(3)} ms/s` : '—';
      const age = (n.age_s != null) ? `${n.age_s.toFixed(1)} s` : 'n/a';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${n.node_id}</td>
        <td><span class="${cls}">${pillText(n.state)}</span></td>
        <td>${centered}</td>
        <td>${off}</td>
        <td>${rate}</td>
        <td>${age}</td>
        <td class="muted small">${n.reason || ''}</td>
      `;
      tb.appendChild(tr);
    });
  }

  function renderLinksTable(links){
    const tb = document.querySelector('#linksTable tbody');
    tb.innerHTML = '';
    if(!links || !links.length){
      tb.innerHTML = '<tr><td colspan="11" class="muted small">Keine Link-Daten.</td></tr>';
      return;
    }

    links.forEach(l=>{
      const statePill = l.state === 'UP' ? '<span class="pill ok">UP</span>' : '<span class="pill bad">DOWN</span>';
      const stable = l.stable_conv ? '<span class="pill ok">stable</span>' : '<span class="pill warn">—</span>';

      const td = (v)=> (v==null || !Number.isFinite(v)) ? '<span class="muted small">—</span>' : (v.toFixed(2) + ' ms');
      const age = (l.age_s==null || !Number.isFinite(l.age_s)) ? '<span class="muted small">—</span>' : (l.age_s.toFixed(1) + ' s');

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="mono">${l.link_id}</td>
        <td>${statePill}</td>
        <td>${td(l.theta_last)}</td>
        <td>${td(l.theta_med)}</td>
        <td>${td(l.rtt_last)}</td>
        <td>${td(l.rtt_med)}</td>
        <td>${td(l.sigma_last)}</td>
        <td>${td(l.sigma_med)}</td>
        <td>${age}</td>
        <td>${l.n ?? 0}</td>
        <td>${stable}</td>
      `;
      tb.appendChild(tr);
    });
  }

  function renderLinksMini(links){
    const tb = document.querySelector('#linksMiniTable tbody');
    tb.innerHTML = '';
    if(!links || !links.length){
      tb.innerHTML = '<tr><td colspan="5" class="muted small">Keine Link-Daten.</td></tr>';
      return;
    }

    // sort: worst sigma med desc, then age
    const sorted = [...links].sort((a,b)=>{
      const sa = (a.sigma_med != null && Number.isFinite(a.sigma_med)) ? a.sigma_med : -1;
      const sb = (b.sigma_med != null && Number.isFinite(b.sigma_med)) ? b.sigma_med : -1;
      return sb - sa;
    }).slice(0, 14);

    sorted.forEach(l=>{
      const st = l.state === 'UP' ? '<span class="pill ok">UP</span>' : '<span class="pill bad">DOWN</span>';
      const sig = (l.sigma_med==null || !Number.isFinite(l.sigma_med)) ? '—' : (l.sigma_med.toFixed(2) + ' ms');
      const rtt = (l.rtt_med==null || !Number.isFinite(l.rtt_med)) ? '—' : (l.rtt_med.toFixed(2) + ' ms');
      const age = (l.age_s==null || !Number.isFinite(l.age_s)) ? '—' : (l.age_s.toFixed(1) + ' s');
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="mono">${l.link_id}</td>
        <td>${st}</td>
        <td>${sig}</td>
        <td>${rtt}</td>
        <td class="muted">${age}</td>
      `;
      tb.appendChild(tr);
    });
  }

  // -------- refresh loop ----------
  let topoCache = null;
  let nodesCache = [];
  let linksCache = [];
  let nodesTSCache = null;

  async function refresh(){
    try{
      const [topo, ov, links, nodesTS] = await Promise.all([
        topoCache ? Promise.resolve(topoCache) : fetchJson('/api/topology'),
        fetchJson('/api/overview'),
        fetchJson('/api/links_table'),
        fetchJson('/api/nodes_timeseries'),
      ]);
      topoCache = topo;

      nodesCache = renderOverview(ov) || [];
      linksCache = (links.links || []);

      renderLinksMini(linksCache);
      renderLinksTable(linksCache);
      renderNodesTable(nodesCache);

      // mesh chart: centered vs absolute
      nodesTSCache = nodesTS;
      updateMeshTimeChart();

      // draw mesh (node states + link sigma map)
      const meshCanvas = document.getElementById('meshCanvas');
      meshCanvas.width = meshCanvas.clientWidth;
      meshCanvas.height = meshCanvas.clientHeight;

      const nodeStates = {};
      nodesCache.forEach(n => nodeStates[n.node_id] = n.state);

      const linkSigmaMap = {};
      linksCache.forEach(l => {
        if(l && l.link_id) linkSigmaMap[l.link_id] = l.sigma_med_conv ?? l.sigma_med ?? l.sigma_last;
      });

      drawMesh(meshCanvas, topo, nodeStates, linkSigmaMap);

      // Debug charts for links
      if(document.body.classList.contains('debugOn')){
        // build pseudo-series from table (not a full timeseries): keep lightweight
        // For real link timeseries, you'd add /api/link_timeseries again.
        // Here we just show a bar-like line with one point (most recent) to avoid noise.
        const thetaSeries = {};
        const sigmaSeries = {};
        const now = Date.now()/1000;
        linksCache.slice(0, 10).forEach((l, i)=>{
          thetaSeries[l.link_id] = [{ t: now, v: (Number.isFinite(l.theta_last)? l.theta_last : null) }];
          sigmaSeries[l.link_id] = [{ t: now, v: (Number.isFinite(l.sigma_last)? l.sigma_last : null) }];
        });
        // convert shape to expected
        const mapTheta = {};
        const mapSigma = {};
        Object.keys(thetaSeries).forEach(k => mapTheta[k] = thetaSeries[k].map(p => ({ t:p.t, theta_ms:p.v })));
        Object.keys(sigmaSeries).forEach(k => mapSigma[k] = sigmaSeries[k].map(p => ({ t:p.t, sigma_ms:p.v })));
        updateLineBySeries(thetaChart, mapTheta, "theta_ms", "");
        updateLineBySeries(sigmaChart, mapSigma, "sigma_ms", "");
      }

      // Heatmap
      const metric = document.getElementById('heatMetricSel').value;
      const hm = await fetchJson('/api/heatmap?metric=' + encodeURIComponent(metric));
      updateHeatmap(hm);

      // Controller (Debug)
      await refreshControllerIfNeeded();

    }catch(e){
      console.error('refresh failed:', e);
    }
  }

  function updateMeshTimeChart(){
    const view = document.getElementById('meshViewSel').value;
    const series = (nodesTSCache && nodesTSCache.series) ? nodesTSCache.series : {};
    // We mapped points as {t, mesh_ms_abs, mesh_ms_centered, offset_ms, offset_centered_ms}
    // Show mesh_ms_* when present; fallback to offset_*.
    const out = {};
    Object.keys(series).forEach(nid=>{
      const pts = series[nid] || [];
      out[nid] = pts.map(p=>{
        let v = null;
        if(view === 'absolute'){
          v = (p.mesh_ms_abs != null) ? p.mesh_ms_abs : (p.offset_ms != null ? p.offset_ms : null);
        } else {
          v = (p.mesh_ms_centered != null) ? p.mesh_ms_centered : (p.offset_centered_ms != null ? p.offset_centered_ms : null);
        }
        return { t:p.t, value:v };
      }).filter(p => p.value != null && Number.isFinite(p.value))
        .map(p => ({ t:p.t, mesh:p.value }));
    });

    // adapt to chart helper format
    const mapped = {};
    Object.keys(out).forEach(nid => mapped[nid] = out[nid].map(p => ({ t:p.t, mesh:p.mesh })));
    updateLineBySeries(meshTimeChart, mapped, "mesh", "Node ");
  }

  async function refreshControllerIfNeeded(){
    if(!document.body.classList.contains('debugOn')) return;

    // populate node selector from overview nodes (once)
    const sel = document.getElementById('ctrlNodeSel');
    if(sel.options.length === 0){
      const ids = nodesCache.map(n => n.node_id).sort();
      ids.forEach(id=>{
        const opt = document.createElement('option');
        opt.value = id;
        opt.textContent = id;
        sel.appendChild(opt);
      });
    }
    const node = sel.value || (sel.options[0]?.value || '');
    const ctrl = await fetchJson('/api/controller?node=' + encodeURIComponent(node));
    updateControllerCharts(ctrl.series || []);
  }

  // -------- init ----------
  window.addEventListener('load', ()=>{
    initCharts();

    document.getElementById('debugToggle').addEventListener('change', ()=>{
      applyDebug();
      // refresh controller tab visibility by toggling class
      // also refresh once to populate controller select if needed
      refresh();
    });

    document.querySelectorAll('.tabbtn').forEach(btn=>{
      btn.addEventListener('click', ()=>{
        const tab = btn.dataset.tab;
        if(tab === 'controller' && !document.body.classList.contains('debugOn')) return;
        setTab(tab);
      });
    });

    document.getElementById('meshViewSel').addEventListener('change', ()=>{
      updateMeshTimeChart();
    });

    document.getElementById('heatMetricSel').addEventListener('change', ()=>{
      // lazy refresh heatmap only
      refresh();
    });

    document.getElementById('ctrlNodeSel').addEventListener('change', ()=>{
      refreshControllerIfNeeded();
    });

    applyDebug();
    refresh();
    setInterval(refresh, 2000);
  });
</script>

</body>
</html>
"""


# -----------------------------
# Boot
# -----------------------------
def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Storage(str(DB_PATH))  # create schema if missing


if __name__ == "__main__":
    ensure_db()
    print("Starting MeshTime Web-UI on http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)
