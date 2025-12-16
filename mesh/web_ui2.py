# -*- coding: utf-8 -*-
# mesh/web_ui.py — MeshTime Dashboard v2 (CLEAN + SLIM)
# Python 3.7 compatible

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

from mesh.storage import Storage  # ensures schema exists

app = Flask(__name__)

# -----------------------------
# Defaults / UI knobs
# -----------------------------
WINDOW_S_DEFAULT = 600.0
BIN_S_DEFAULT = 0.5

MAX_NODE_POINTS_DEFAULT = 20000
MAX_LINK_POINTS_DEFAULT = 40000
MAX_CTRL_POINTS_DEFAULT = 20000

# Overview heuristics
CONV_WINDOW_S = 30.0
MIN_SAMPLES_WARMUP = 8
FRESH_MIN_S = 3.0
FRESH_MULT = 6.0
THRESH_LINK_SIGMA_MED_MS = 2.0  # "green" if median sigma <= this
K_STABLE_LINKS = 1


# -----------------------------
# Helpers
# -----------------------------
def _json_error(msg: str, status: int = 500):
    return make_response(jsonify({"error": str(msg)}), int(status))


def get_conn() -> sqlite3.Connection:
    c = sqlite3.connect(str(DB_PATH), isolation_level=None)
    c.row_factory = sqlite3.Row
    # keep it fast + robust
    try:
        c.execute("PRAGMA journal_mode=WAL;")
        c.execute("PRAGMA synchronous=NORMAL;")
        c.execute("PRAGMA busy_timeout=2000;")
    except Exception:
        pass
    return c


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    try:
        r = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=? LIMIT 1", (table,)
        ).fetchone()
        return bool(r)
    except Exception:
        return False


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


def p95(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return float(xs[0])
    k = int(round(0.95 * (len(xs) - 1)))
    k = max(0, min(len(xs) - 1, k))
    return float(xs[k])


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
            if node_id < neigh:
                links.append({"source": node_id, "target": neigh})

    return {"nodes": nodes, "links": links}


# -----------------------------
# Data source selection
# -----------------------------
def _table_has_rows(conn, table: str) -> bool:
    try:
        r = conn.execute("SELECT 1 FROM %s LIMIT 1" % table).fetchone()
        return bool(r)
    except Exception:
        return False

def _pick_sources(conn):
    return {
        "mesh": "mesh_clock" if (_table_exists(conn,"mesh_clock") and _table_has_rows(conn,"mesh_clock")) else "ntp_reference",
        "link": "obs_link" if (_table_exists(conn,"obs_link") and _table_has_rows(conn,"obs_link")) else "ntp_reference",
        "ctrl": "diag_controller" if (_table_exists(conn,"diag_controller") and _table_has_rows(conn,"diag_controller")) else "ntp_reference",
        "kalman": "diag_kalman" if (_table_exists(conn,"diag_kalman") and _table_has_rows(conn,"diag_kalman")) else "",
    }



# -----------------------------
# Fetchers
# -----------------------------
def fetch_mesh_clock_rows(conn: sqlite3.Connection, window_s: float, limit: int) -> List[sqlite3.Row]:
    cols = _table_cols(conn, "mesh_clock")
    need = {"created_at_s", "node_id", "t_mesh_s"}
    if not need.issubset(cols):
        return []
    cutoff = time.time() - float(window_s)
    q = """
        SELECT created_at_s, node_id, t_mesh_s
        FROM mesh_clock
        WHERE created_at_s >= ?
        ORDER BY created_at_s ASC
        LIMIT ?
    """
    return conn.execute(q, (cutoff, int(limit))).fetchall()


def fetch_ntp_node_rows(conn: sqlite3.Connection, window_s: float, limit: int) -> List[sqlite3.Row]:
    cols = _table_cols(conn, "ntp_reference")
    need = {"node_id", "created_at", "t_mesh"}
    if not need.issubset(cols):
        return []
    cutoff = time.time() - float(window_s)

    # node rows: peer_id NULL if column exists
    peer_filter = ""
    if "peer_id" in cols:
        peer_filter = "AND (peer_id IS NULL OR peer_id='')"

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


def fetch_obs_link_rows(conn: sqlite3.Connection, window_s: float, limit: int) -> List[sqlite3.Row]:
    cols = _table_cols(conn, "obs_link")
    need = {"created_at_s", "node_id", "peer_id"}
    if not need.issubset(cols):
        return []
    cutoff = time.time() - float(window_s)

    sel = ["created_at_s", "node_id", "peer_id"]
    for c in ("theta_ms", "rtt_ms", "sigma_ms", "accepted", "weight", "reject_reason"):
        if c in cols:
            sel.append(c)

    q = """
        SELECT {sel}
        FROM obs_link
        WHERE created_at_s >= ?
        ORDER BY created_at_s ASC
        LIMIT ?
    """.format(sel=", ".join(sel))

    return conn.execute(q, (cutoff, int(limit))).fetchall()


def fetch_ntp_link_rows(conn, window_s, limit):
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
          AND peer_id != ''
        ORDER BY created_at ASC
        LIMIT ?
    """.format(sel=", ".join(sel))

    return conn.execute(q, (cutoff, int(limit))).fetchall()


def fetch_diag_controller_rows(conn: sqlite3.Connection, window_s: float, limit: int) -> List[sqlite3.Row]:
    cols = _table_cols(conn, "diag_controller")
    need = {"created_at_s", "node_id"}
    if not need.issubset(cols):
        return []
    cutoff = time.time() - float(window_s)

    sel = ["created_at_s", "node_id"]
    for c in ("dt_s", "delta_desired_ms", "delta_applied_ms", "slew_clipped", "max_slew_ms_s", "eff_eta"):
        if c in cols:
            sel.append(c)

    q = """
        SELECT {sel}
        FROM diag_controller
        WHERE created_at_s >= ?
        ORDER BY created_at_s ASC
        LIMIT ?
    """.format(sel=", ".join(sel))
    return conn.execute(q, (cutoff, int(limit))).fetchall()


def fetch_ntp_ctrl_rows(conn, window_s, limit):
    cols = _table_cols(conn, "ntp_reference")
    if not {"node_id", "created_at"}.issubset(cols):
        return []

    ctrl_cols = [c for c in ("delta_desired_ms", "delta_applied_ms", "dt_s", "slew_clipped") if c in cols]
    if not ctrl_cols:
        return []

    cutoff = time.time() - float(window_s)
    nonnull = " OR ".join(["%s IS NOT NULL" % c for c in ctrl_cols])

    sel = ["node_id", "created_at"] + ctrl_cols

    peer_guard = ""
    if "peer_id" in cols:
        peer_guard = "AND (peer_id IS NOT NULL AND peer_id != '')"

    q = """
        SELECT {sel}
        FROM ntp_reference
        WHERE created_at >= ?
          {peer_guard}
          AND ({nonnull})
        ORDER BY created_at ASC
        LIMIT ?
    """.format(sel=", ".join(sel), peer_guard=peer_guard, nonnull=nonnull)

    return conn.execute(q, (cutoff, int(limit))).fetchall()



def fetch_diag_kalman_rows(conn: sqlite3.Connection, window_s: float, limit: int) -> List[sqlite3.Row]:
    cols = _table_cols(conn, "diag_kalman")
    need = {"created_at_s", "node_id"}
    if not need.issubset(cols):
        return []
    cutoff = time.time() - float(window_s)

    sel = ["created_at_s", "node_id"]
    for c in ("n_meas", "innov_med_ms", "innov_p95_ms", "nis_med", "nis_p95", "x_offset_ms", "x_drift_ppm",
              "p_offset_ms2", "p_drift_ppm2", "r_eff_ms2"):
        if c in cols:
            sel.append(c)

    q = """
        SELECT {sel}
        FROM diag_kalman
        WHERE created_at_s >= ?
        ORDER BY created_at_s ASC
        LIMIT ?
    """.format(sel=", ".join(sel))
    return conn.execute(q, (cutoff, int(limit))).fetchall()


# -----------------------------
# Build diagnostics payloads
# -----------------------------
def build_mesh_diag(conn: sqlite3.Connection, window_s: float, max_points: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      meta
      mesh_raw[node]    = [{t_wall, mesh_err_ms}]
      mesh_smooth[node] = [{t_wall, mesh_err_ms}]
      step_raw[node]    = [{t_wall, step_ms}]
      step_smooth[node] = [{t_wall, step_ms}]
      pairs_raw[A-B]    = [{t_wall, delta_ms}]
      legacy (optional from ntp_reference): offset_ms + err_ms series
    """

    sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
    beacon_period = _beacon_period_s(cfg)

    # TTL: what counts as "fresh" for consensus at an event time
    ttl_s = sync.get("ui_ttl_s", None)
    if ttl_s is None:
        ttl_s = max(2.0, 3.0 * beacon_period)
    try:
        ttl_s = float(ttl_s)
    except Exception:
        ttl_s = max(2.0, 3.0 * beacon_period)

    # EWMA smoothing time constant
    tau_s = sync.get("ui_smooth_tau_s", None)
    if tau_s is None:
        tau_s = max(1.0, 2.0 * beacon_period)
    try:
        tau_s = float(tau_s)
    except Exception:
        tau_s = max(1.0, 2.0 * beacon_period)

    src = _pick_sources(conn)["mesh"]

    events: List[Tuple[float, str, float]] = []  # (t_wall, node, mesh_offset_ms-like)
    nodes: Set[str] = set()

    if src == "mesh_clock":
        rows = fetch_mesh_clock_rows(conn, window_s, max_points)
        for r in rows:
            t = _f(r["created_at_s"])
            nid = str(r["node_id"])
            tm = _f(r["t_mesh_s"])
            if t is None or tm is None or not nid:
                continue
            # "mesh_offset_ms" definition: (t_mesh - created_at)*1000
            mo = (float(tm) - float(t)) * 1000.0
            events.append((float(t), nid, float(mo)))
            nodes.add(nid)
    else:
        # Fallback (legacy): ntp_reference node rows
        rows = fetch_ntp_node_rows(conn, window_s, max_points)
        # Infer t_mesh scale (seconds vs ms/us) using delta ratios
        # If it's already seconds, scale=1. If looks like ms, scale=1e-3, etc.
        scale = infer_tmesh_to_seconds(rows)
        for r in rows:
            t = _f(r["created_at"])
            nid = str(r["node_id"])
            tm = _f(r["t_mesh"])
            if t is None or tm is None or not nid:
                continue
            mo = (float(tm) * float(scale) - float(t)) * 1000.0
            events.append((float(t), nid, float(mo)))
            nodes.add(nid)

    events.sort(key=lambda x: x[0])
    nodes_sorted = sorted(nodes)

    if not events or not nodes_sorted:
        return {
            "meta": {
                "note": "no mesh samples",
                "source": src,
                "window_s": float(window_s),
                "ttl_s": float(ttl_s),
                "smooth_tau_s": float(tau_s),
            },
            "mesh_raw": {},
            "mesh_smooth": {},
            "step_raw": {},
            "step_smooth": {},
            "pairs_raw": {},
            "legacy": {},
        }

    # Event-based consensus (controller-like): append point only for the node that produced the event
    last_val: Dict[str, float] = {}
    last_time: Dict[str, float] = {}

    mesh_raw: Dict[str, List[Dict[str, float]]] = {n: [] for n in nodes_sorted}
    pairs_raw: Dict[str, List[Dict[str, float]]] = {}

    for (t, n, mo) in events:
        last_val[n] = mo
        last_time[n] = t

        fresh: Dict[str, float] = {}
        for nn in nodes_sorted:
            lt = last_time.get(nn)
            lv = last_val.get(nn)
            if lt is None or lv is None:
                continue
            if (t - lt) <= ttl_s:
                fresh[nn] = lv

        if len(fresh) < 2:
            continue

        cons = median(list(fresh.values()))
        if cons is None:
            continue

        e_n = float(last_val[n] - float(cons))
        mesh_raw[n].append({"t_wall": float(t), "mesh_err_ms": float(e_n)})

        # Pair series need full eps map
        ns = sorted(fresh.keys())
        eps_map = {nn: float(fresh[nn] - float(cons)) for nn in ns}
        for i in range(len(ns)):
            for j in range(i + 1, len(ns)):
                pid = "%s-%s" % (ns[i], ns[j])
                pairs_raw.setdefault(pid, []).append(
                    {"t_wall": float(t), "delta_ms": float(eps_map[ns[i]] - eps_map[ns[j]])}
                )

    mesh_raw = {n: pts for (n, pts) in mesh_raw.items() if pts}

    def ewma_time_smooth(series: List[Dict[str, float]], tau: float) -> List[Dict[str, float]]:
        if not series:
            return []
        y = None
        t_prev = None
        out = []
        for p in series:
            tt = float(p["t_wall"])
            x = float(p["mesh_err_ms"])
            if y is None:
                y = x
                t_prev = tt
            else:
                dt = max(0.0, tt - float(t_prev))
                a = 1.0 - math.exp(-dt / max(1e-6, tau))
                y = (1.0 - a) * float(y) + a * x
                t_prev = tt
            out.append({"t_wall": float(tt), "mesh_err_ms": float(y)})
        return out

    mesh_smooth: Dict[str, List[Dict[str, float]]] = {}
    for n, pts in mesh_raw.items():
        mesh_smooth[n] = ewma_time_smooth(pts, tau_s)

    def steps_from_mesh(mesh_series: Dict[str, List[Dict[str, float]]]) -> Dict[str, List[Dict[str, float]]]:
        out = {}
        for n, pts in mesh_series.items():
            prev = None
            s = []
            for p in pts:
                cur = float(p["mesh_err_ms"])
                d = 0.0 if prev is None else (cur - float(prev))
                s.append({"t_wall": float(p["t_wall"]), "step_ms": float(d)})
                prev = cur
            out[n] = s
        return out

    step_raw = steps_from_mesh(mesh_raw)
    step_smooth = steps_from_mesh(mesh_smooth)

    # Legacy series (offset / err_mesh_vs_wall) if available in ntp_reference
    legacy = {"offset_ms": {}, "err_ms": {}}
    if _table_exists(conn, "ntp_reference"):
        cols = _table_cols(conn, "ntp_reference")
        if {"node_id", "created_at"}.issubset(cols) and ("offset" in cols or "err_mesh_vs_wall" in cols):
            rows2 = fetch_ntp_node_rows(conn, window_s, min(max_points, MAX_NODE_POINTS_DEFAULT))
            by_node = {}
            for r in rows2:
                nid = str(r["node_id"])
                t = _f(r["created_at"])
                if not nid or t is None:
                    continue
                obj = {"t_wall": float(t)}
                if "offset" in r.keys():
                    v = _f(r["offset"])
                    if v is not None:
                        obj["offset_ms"] = float(v) * 1000.0
                if "err_mesh_vs_wall" in r.keys():
                    v = _f(r["err_mesh_vs_wall"])
                    if v is not None:
                        obj["err_ms"] = float(v) * 1000.0
                by_node.setdefault(nid, []).append(obj)
            for nid, pts in by_node.items():
                pts.sort(key=lambda p: p["t_wall"])
                legacy["offset_ms"][nid] = [{"t_wall": p["t_wall"], "offset_ms": p["offset_ms"]} for p in pts if "offset_ms" in p]
                legacy["err_ms"][nid] = [{"t_wall": p["t_wall"], "err_ms": p["err_ms"]} for p in pts if "err_ms" in p]

    return {
        "meta": {
            "source": src,
            "window_s": float(window_s),
            "ttl_s": float(ttl_s),
            "smooth_tau_s": float(tau_s),
            "nodes": nodes_sorted,
            "x_axis": "created_at_s",
            "mode": "event+ttl",
        },
        "mesh_raw": mesh_raw,
        "mesh_smooth": mesh_smooth,
        "step_raw": step_raw,
        "step_smooth": step_smooth,
        "pairs_raw": pairs_raw,
        "legacy": legacy,
    }


def infer_tmesh_to_seconds(rows: List[sqlite3.Row]) -> float:
    """
    For ntp_reference fallback:
    infer scale s.t. t_mesh_seconds = t_mesh_raw * scale
    """
    by_node: Dict[str, List[Tuple[float, float]]] = {}
    for r in rows:
        try:
            nid = str(r["node_id"])
            tc = _f(r["created_at"])
            tm = _f(r["t_mesh"])
        except Exception:
            continue
        if nid and tc is not None and tm is not None:
            by_node.setdefault(nid, []).append((tc, tm))

    ratios: List[float] = []
    for pts in by_node.values():
        pts.sort(key=lambda x: x[0])
        for i in range(1, len(pts)):
            dt = float(pts[i][0] - pts[i - 1][0])
            dm = float(pts[i][1] - pts[i - 1][1])
            if dt <= 1e-6 or dm <= 0:
                continue
            ratios.append(dm / dt)

    r_med = median(ratios)
    if r_med is None or r_med <= 0:
        return 1.0

    inv = 1.0 / float(r_med)
    for snap in (1.0, 1e-3, 1e-6):
        try:
            if abs(math.log10(inv) - math.log10(snap)) < 0.25:
                return float(snap)
        except Exception:
            pass
    return float(inv)


def build_link_diag(conn: sqlite3.Connection, window_s: float, bin_s: float, max_points: int) -> Dict[str, Any]:
    """
    Returns:
      meta
      links[link_id] = [{t_wall, theta_ms?, rtt_ms?, sigma_ms?}]
      latest_sigma[link_id] = sigma_ms
    """
    if bin_s <= 0:
        bin_s = BIN_S_DEFAULT

    src = _pick_sources(conn)["link"]
    bins: Dict[int, Dict[str, Dict[str, List[float]]]] = {}
    latest_sigma: Dict[str, float] = {}

    if src == "obs_link":
        rows = fetch_obs_link_rows(conn, window_s, max_points)
        for r in rows:
            t = _f(r["created_at_s"])
            if t is None:
                continue
            lid = "%s->%s" % (str(r["node_id"]), str(r["peer_id"]))
            idx = int(math.floor(float(t) / float(bin_s)))
            d = bins.setdefault(idx, {}).setdefault(lid, {"theta_ms": [], "rtt_ms": [], "sigma_ms": []})
            for k in ("theta_ms", "rtt_ms", "sigma_ms"):
                if k in r.keys():
                    v = _f(r[k])
                    if v is not None:
                        d[k].append(float(v))
    else:
        rows = fetch_ntp_link_rows(conn, window_s, max_points)
        for r in rows:
            t = _f(r["created_at"])
            if t is None:
                continue
            lid = "%s->%s" % (str(r["node_id"]), str(r["peer_id"]))
            idx = int(math.floor(float(t) / float(bin_s)))
            d = bins.setdefault(idx, {}).setdefault(lid, {"theta_ms": [], "rtt_ms": [], "sigma_ms": []})
            for k in ("theta_ms", "rtt_ms", "sigma_ms"):
                if k in r.keys():
                    v = _f(r[k])
                    if v is not None:
                        d[k].append(float(v))

    out: Dict[str, List[Dict[str, float]]] = {}

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

    return {
        "meta": {"source": src, "window_s": float(window_s), "bin_s": float(bin_s), "x_axis": "created_at_s"},
        "links": out,
        "latest_sigma": latest_sigma,
    }


def _maybe_scale_controller_deltas(vals: List[float]) -> float:
    """
    In your earlier history: some deltas were in seconds even though named *_ms.
    Heuristic: if median(|delta|) < 1.0 => treat as seconds => scale 1000 to ms
    """
    m = median([abs(v) for v in vals if v is not None])
    if m is None:
        return 1.0
    return 1000.0 if float(m) < 1.0 else 1.0


def build_controller_diag(conn: sqlite3.Connection, window_s: float, max_points: int) -> Dict[str, Any]:
    """
    Returns:
      meta
      controller[node] = [{t_wall, delta_desired_ms, delta_applied_ms, dt_s, slew_clipped, eff_eta?}]
      kalman[node]     = [{t_wall, innov_med_ms, nis_med, x_drift_ppm, ...}]   (if diag_kalman exists)
    """
    sources = _pick_sources(conn)
    src = sources["ctrl"]

    out: Dict[str, List[Dict[str, Any]]] = {}
    scale_to_ms = 1.0

    if src == "diag_controller":
        rows = fetch_diag_controller_rows(conn, window_s, max_points)

        # detect if deltas need scaling
        raw = []
        for r in rows:
            for k in ("delta_desired_ms", "delta_applied_ms"):
                if k in r.keys():
                    v = _f(r[k])
                    if v is not None:
                        raw.append(float(v))
        scale_to_ms = _maybe_scale_controller_deltas(raw)

        for r in rows:
            nid = str(r["node_id"])
            t = _f(r["created_at_s"])
            if not nid or t is None:
                continue
            obj: Dict[str, Any] = {"t_wall": float(t)}

            if "delta_desired_ms" in r.keys():
                v = _f(r["delta_desired_ms"])
                if v is not None:
                    obj["delta_desired_ms"] = float(v) * float(scale_to_ms)

            if "delta_applied_ms" in r.keys():
                v = _f(r["delta_applied_ms"])
                if v is not None:
                    obj["delta_applied_ms"] = float(v) * float(scale_to_ms)

            if "dt_s" in r.keys():
                v = _f(r["dt_s"])
                if v is not None:
                    obj["dt_s"] = float(v)

            if "slew_clipped" in r.keys() and r["slew_clipped"] is not None:
                try:
                    obj["slew_clipped"] = 1 if int(r["slew_clipped"]) else 0
                except Exception:
                    pass

            if "eff_eta" in r.keys():
                v = _f(r["eff_eta"])
                if v is not None:
                    obj["eff_eta"] = float(v)

            out.setdefault(nid, []).append(obj)

    else:
        # fallback: ntp_reference debug cols
        rows = fetch_ntp_ctrl_rows(conn, window_s, max_points)
        raw = []
        for r in rows:
            for k in ("delta_desired_ms", "delta_applied_ms"):
                if k in r.keys():
                    v = _f(r[k])
                    if v is not None:
                        raw.append(float(v))
        scale_to_ms = _maybe_scale_controller_deltas(raw)

        for r in rows:
            nid = str(r["node_id"])
            t = _f(r["created_at"])
            if not nid or t is None:
                continue
            obj: Dict[str, Any] = {"t_wall": float(t)}
            for k in ("delta_desired_ms", "delta_applied_ms"):
                if k in r.keys():
                    v = _f(r[k])
                    if v is not None:
                        obj[k] = float(v) * float(scale_to_ms)
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

    # kalman diag is optional but nice (and cheap)
    kalman_out: Dict[str, List[Dict[str, Any]]] = {}
    if sources.get("kalman"):
        rowsk = fetch_diag_kalman_rows(conn, window_s, max_points)
        for r in rowsk:
            nid = str(r["node_id"])
            t = _f(r["created_at_s"])
            if not nid or t is None:
                continue
            obj: Dict[str, Any] = {"t_wall": float(t)}
            for k in ("n_meas", "innov_med_ms", "innov_p95_ms", "nis_med", "nis_p95", "x_offset_ms",
                      "x_drift_ppm", "p_offset_ms2", "p_drift_ppm2", "r_eff_ms2"):
                if k in r.keys():
                    v = _f(r[k])
                    if v is not None:
                        obj[k] = float(v)
            kalman_out.setdefault(nid, []).append(obj)
        for nid in kalman_out.keys():
            kalman_out[nid].sort(key=lambda p: p["t_wall"])

    return {
        "meta": {"source": src, "window_s": float(window_s), "x_axis": "created_at_s", "delta_scale_to_ms": float(scale_to_ms)},
        "controller": out,
        "kalman": kalman_out,
    }


def compute_overview(conn: sqlite3.Connection, window_s: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very pragmatic, deterministic overview:
      - node freshness (last sample age)
      - warmup (min samples in CONV_WINDOW)
      - link stability via median sigma (conv window)
      - mesh_err_now relative to median of current offsets (if we have >=2 fresh)
    """
    now = time.time()
    fresh_thr = _fresh_thresh_s(cfg)
    conv_cut = float(now) - float(CONV_WINDOW_S)

    sources = _pick_sources(conn)

    # -------- latest mesh_offset per node (fresh)
    latest_t: Dict[str, float] = {}
    latest_mo: Dict[str, float] = {}  # mesh_offset_ms-like value per node

    if sources["mesh"] == "mesh_clock":
        rows = fetch_mesh_clock_rows(conn, max(window_s, CONV_WINDOW_S), MAX_NODE_POINTS_DEFAULT)
        for r in rows:
            t = _f(r["created_at_s"])
            nid = str(r["node_id"])
            tm = _f(r["t_mesh_s"])
            if t is None or tm is None or not nid:
                continue
            mo = (float(tm) - float(t)) * 1000.0
            latest_t[nid] = float(t)
            latest_mo[nid] = float(mo)
    else:
        rows = fetch_ntp_node_rows(conn, max(window_s, CONV_WINDOW_S), MAX_NODE_POINTS_DEFAULT)
        scale = infer_tmesh_to_seconds(rows)
        for r in rows:
            t = _f(r["created_at"])
            nid = str(r["node_id"])
            tm = _f(r["t_mesh"])
            if t is None or tm is None or not nid:
                continue
            mo = (float(tm) * float(scale) - float(t)) * 1000.0
            latest_t[nid] = float(t)
            latest_mo[nid] = float(mo)

    # consensus now (median of fresh nodes only)
    fresh_offsets = []
    for nid, t in latest_t.items():
        if (now - float(t)) <= float(fresh_thr):
            if nid in latest_mo:
                fresh_offsets.append(float(latest_mo[nid]))
    cons_now = median(fresh_offsets)

    # -------- link sigma med in conv window
    sigs_by_link: Dict[str, List[float]] = {}
    last_seen_link: Dict[str, float] = {}

    if sources["link"] == "obs_link":
        lrows = fetch_obs_link_rows(conn, max(window_s, CONV_WINDOW_S), MAX_LINK_POINTS_DEFAULT)
        for r in lrows:
            t = _f(r["created_at_s"])
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
    else:
        lrows = fetch_ntp_link_rows(conn, max(window_s, CONV_WINDOW_S), MAX_LINK_POINTS_DEFAULT)
        for r in lrows:
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

    # -------- warmup sample counts in conv window (per node)
    sample_count_conv: Dict[str, int] = {}
    if sources["mesh"] == "mesh_clock":
        rows2 = fetch_mesh_clock_rows(conn, max(window_s, CONV_WINDOW_S), MAX_NODE_POINTS_DEFAULT)
        for r in rows2:
            t = _f(r["created_at_s"])
            if t is None or float(t) < conv_cut:
                continue
            nid = str(r["node_id"])
            sample_count_conv[nid] = sample_count_conv.get(nid, 0) + 1
    else:
        rows2 = fetch_ntp_node_rows(conn, max(window_s, CONV_WINDOW_S), MAX_NODE_POINTS_DEFAULT)
        for r in rows2:
            t = _f(r["created_at"])
            if t is None or float(t) < conv_cut:
                continue
            nid = str(r["node_id"])
            sample_count_conv[nid] = sample_count_conv.get(nid, 0) + 1

    # -------- node statuses
    nodes_out: List[Dict[str, Any]] = []
    all_nodes = sorted(set(list(latest_t.keys())))

    for nid in all_nodes:
        t_last = latest_t.get(nid)
        age_s = float(now - float(t_last)) if t_last is not None else None
        fresh_ok = (age_s is not None and float(age_s) <= float(fresh_thr))
        warm_ok = (sample_count_conv.get(nid, 0) >= int(MIN_SAMPLES_WARMUP))

        mesh_err_now_ms = None
        if cons_now is not None and nid in latest_mo and fresh_ok:
            mesh_err_now_ms = float(latest_mo[nid] - float(cons_now))

        # stable outgoing links (fresh)
        stable_links = 0
        considered = 0
        for lid, sig in link_sigma_med.items():
            if not lid.startswith(nid + "->"):
                continue
            ls = last_seen_link.get(lid)
            if ls is None:
                continue
            if float(now - float(ls)) > float(fresh_thr):
                continue
            considered += 1
            if float(sig) <= float(THRESH_LINK_SIGMA_MED_MS):
                stable_links += 1

        if not warm_ok:
            state, reason = "YELLOW", "warming up"
        elif not fresh_ok:
            state, reason = "RED", "stale data (age %.1fs)" % (age_s if age_s is not None else -1.0)
        else:
            if considered == 0:
                state, reason = "YELLOW", "no fresh link sigma"
            elif stable_links >= int(K_STABLE_LINKS):
                state, reason = "GREEN", "converged"
            else:
                state, reason = "YELLOW", "unstable links"

        nodes_out.append(
            {
                "node_id": nid,
                "state": state,
                "reason": reason,
                "last_seen_utc": _utc(t_last),
                "age_s": age_s,
                "mesh_err_now_ms": mesh_err_now_ms,
                "stable_links": stable_links,
                "links_considered": considered,
                "k_stable_links": int(K_STABLE_LINKS),
            }
        )

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
            "thresholds": {
                "fresh_s": float(fresh_thr),
                "link_sigma_med_ms": float(THRESH_LINK_SIGMA_MED_MS),
                "warmup_min_samples": int(MIN_SAMPLES_WARMUP),
                "k_stable_links": int(K_STABLE_LINKS),
            },
            "sources": sources,
        },
        "nodes": nodes_out,
        "link_sigma_med": link_sigma_med,
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
        return _json_error("/api/topology failed: %s" % e, 500)


@app.route("/api/overview")
def api_overview():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}
        window_s = float(sync.get("ui_window_s", WINDOW_S_DEFAULT))
        with get_conn() as conn:
            return jsonify(compute_overview(conn, window_s, cfg))
    except Exception as e:
        return _json_error("/api/overview failed: %s" % e, 500)


@app.route("/api/mesh_diag")
def api_mesh_diag():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}

        window_s = float(request.args.get("window_s", sync.get("ui_window_s", WINDOW_S_DEFAULT)))
        max_points = int(request.args.get("max_points", sync.get("ui_max_points", MAX_NODE_POINTS_DEFAULT)))

        with get_conn() as conn:
            return jsonify(build_mesh_diag(conn, window_s, max_points, cfg))
    except Exception as e:
        return _json_error("/api/mesh_diag failed: %s" % e, 500)


@app.route("/api/link_diag")
def api_link_diag():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}

        window_s = float(request.args.get("window_s", sync.get("ui_window_s", WINDOW_S_DEFAULT)))
        bin_s = float(request.args.get("bin_s", sync.get("ui_bin_s", BIN_S_DEFAULT)))
        max_points = int(request.args.get("max_points", sync.get("ui_link_max_points", MAX_LINK_POINTS_DEFAULT)))

        with get_conn() as conn:
            return jsonify(build_link_diag(conn, window_s, bin_s, max_points))
    except Exception as e:
        return _json_error("/api/link_diag failed: %s" % e, 500)


@app.route("/api/controller_diag")
def api_controller_diag():
    try:
        cfg = load_config()
        sync = (cfg.get("sync", {}) or {}) if isinstance(cfg, dict) else {}

        window_s = float(request.args.get("window_s", sync.get("ui_window_s", WINDOW_S_DEFAULT)))
        max_points = int(request.args.get("max_points", sync.get("ui_ctrl_max_points", MAX_CTRL_POINTS_DEFAULT)))

        with get_conn() as conn:
            return jsonify(build_controller_diag(conn, window_s, max_points))
    except Exception as e:
        return _json_error("/api/controller_diag failed: %s" % e, 500)


# -----------------------------
# Boilerplate
# -----------------------------
def ensure_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    Storage(str(DB_PATH))


# -----------------------------
# Template (Single Page, slim)
# -----------------------------
TEMPLATE = r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8">
  <title>MeshTime Dashboard v2</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; margin:0; padding: 1.2rem; background:#0f0f10; color:#eee; }
    h1,h2,h3 { margin:0; }
    .sub { margin-top:0.35rem; opacity:0.72; font-size:0.88rem; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; }
    .card { background:#171718; border-radius:14px; padding:1rem 1.1rem; box-shadow: 0 0 0 1px rgba(255,255,255,0.05); }
    .row { display:flex; gap:0.75rem; flex-wrap:wrap; align-items:center; }
    .sp { justify-content:space-between; }
    .pill { display:inline-block; padding:0.12rem 0.55rem; border-radius:999px; font-size:0.75rem; font-weight:750; }
    .ok   { background: rgba(46,204,113,0.14); color:#2ecc71; }
    .warn { background: rgba(241,196,15,0.14); color:#f1c40f; }
    .bad  { background: rgba(231,76,60,0.14); color:#e74c3c; }
    .grid { display:grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 2fr); gap: 1rem; margin-top:1rem; align-items:start; }
    @media (max-width: 1100px) { .grid { grid-template-columns: minmax(0,1fr); } }
    table { width:100%; border-collapse: collapse; font-size:0.92rem; margin-top: 0.6rem; }
    th, td { padding: 0.35rem 0.5rem; text-align:left; vertical-align:top; }
    th { border-bottom: 1px solid rgba(255,255,255,0.15); font-weight:750; }
    tr:nth-child(even) td { background: rgba(255,255,255,0.02); }
    select {
      background:#101010; color:#eee; border: 1px solid rgba(255,255,255,0.18);
      border-radius: 10px; padding: 0.35rem 0.55rem;
    }
    canvas { max-width:100%; }
    .small { font-size:0.85rem; opacity:0.75; }
    .hidden { display:none !important; }
  </style>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
</head>

<body>
  <div class="row sp">
    <div>
      <h1>MeshTime Dashboard v2</h1>
      <div class="sub">DB: <span class="mono">{{ db_path }}</span> · X-Achse: <b>created_at_s (Sink)</b></div>
    </div>

    <div class="row">
      <div class="card" style="padding:0.55rem 0.85rem;">
        <label class="small">
          <input id="debugToggle" type="checkbox" style="transform:scale(1.2); margin-right:0.5rem;">
          Debug (Controller/Kalman)
        </label>
      </div>

      <div class="card" style="padding:0.55rem 0.85rem;">
        <span class="small">Style:</span>
        <select id="meshStyle">
          <option value="raw" selected>raw</option>
          <option value="smooth">smooth</option>
        </select>
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
        <span class="small">Ctrl Node:</span>
        <select id="ctrlNode"></select>
      </div>
    </div>

    <div class="sub" id="thrLine" style="margin-top:0.6rem;">…</div>

    <table id="statusTable">
      <thead>
        <tr>
          <th>Node</th>
          <th>State</th>
          <th>mesh_err_now</th>
          <th>age</th>
          <th>stable_links</th>
          <th>reason</th>
        </tr>
      </thead>
      <tbody><tr><td colspan="6" class="small">lade…</td></tr></tbody>
    </table>
  </div>

  <div class="grid">
    <div class="card">
      <h2 style="font-size:1.05rem;">Centered Mesh Error ε(node,t) (ms)</h2>
      <div class="sub">
        ε basiert auf <span class="mono">(t_mesh - created_at)*1000</span>, pro Event gegen Median der frischen Nodes zentriert.
      </div>
      <canvas id="meshChart" height="170"></canvas>
      <div style="margin-top:0.75rem;">
        <h3 class="small">Pairwise: ε_i − ε_j (ms)</h3>
        <canvas id="pairChart" height="150"></canvas>
      </div>
      <div style="margin-top:0.75rem;">
        <h3 class="small">Step: Δε pro Node (ms)</h3>
        <canvas id="stepChart" height="150"></canvas>
      </div>

      <div style="margin-top:0.75rem;">
        <h3 class="small">Legacy (optional): offset / err_mesh_vs_wall (ms)</h3>
        <canvas id="legacyOffset" height="130"></canvas>
        <canvas id="legacyErr" height="130" style="margin-top:0.6rem;"></canvas>
      </div>
    </div>

    <div>
      <div class="card">
        <h2 style="font-size:1.05rem;">Link Quality</h2>
        <div class="sub">Link-ID ist <span class="mono">A-&gt;B</span> (gerichtet).</div>
        <canvas id="sigmaBar" height="150"></canvas>
        <div class="row" style="margin-top:0.75rem;">
          <div style="flex:1;">
            <h3 class="small">θ (ms)</h3>
            <canvas id="thetaChart" height="140"></canvas>
          </div>
          <div style="flex:1;">
            <h3 class="small">RTT (ms)</h3>
            <canvas id="rttChart" height="140"></canvas>
          </div>
        </div>
      </div>

      <div class="card debugOnly" style="margin-top:1rem;">
        <h2 style="font-size:1.05rem;">Controller (Debug)</h2>
        <canvas id="deltaApplied" height="150"></canvas>
        <canvas id="dtChart" height="130" style="margin-top:0.6rem;"></canvas>
      </div>

      <div class="card debugOnly" style="margin-top:1rem;">
        <h2 style="font-size:1.05rem;">Kalman (Debug)</h2>
        <canvas id="nisChart" height="150"></canvas>
      </div>
    </div>
  </div>

<script>
  const colors = ['#2ecc71','#3498db','#f1c40f','#e74c3c','#9b59b6','#1abc9c','#e67e22'];

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

  function updateLine(chart, seriesMap, field, labelPrefix=''){
    const ids = Object.keys(seriesMap || {}).sort();
    chart.data.datasets = [];
    ids.forEach((id, idx)=>{
      const c = colors[idx % colors.length];
      const pts = (seriesMap[id] || [])
        .filter(p => p && p.t_wall != null && p[field] != null && Number.isFinite(p[field]))
        .map(p => ({ x: new Date(p.t_wall * 1000), y: p[field] }));
      chart.data.datasets.push({ label: `${labelPrefix}${id}`, data: pts, borderColor: c, borderWidth: 1.6 });
    });
    chart.update();
  }

  function updatePairs(chart, pairs){
    const ids = Object.keys(pairs || {}).sort();
    chart.data.datasets = [];
    ids.forEach((id, idx)=>{
      const c = colors[idx % colors.length];
      const pts = (pairs[id] || [])
        .filter(p => p && p.t_wall != null && p.delta_ms != null && Number.isFinite(p.delta_ms))
        .map(p => ({ x: new Date(p.t_wall * 1000), y: p.delta_ms }));
      chart.data.datasets.push({ label: id, data: pts, borderColor: c, borderWidth: 1.6 });
    });
    chart.update();
  }

  function updateSigmaBar(chart, latest){
    const ids = Object.keys(latest||{}).sort();
    chart.data.labels = ids;
    chart.data.datasets[0].data = ids.map(id => latest[id]);
    chart.update();
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
      const age = (n.age_s==null) ? 'n/a' : `${n.age_s.toFixed(1)} s`;
      const sl  = `${n.stable_links ?? 0}/${n.k_stable_links ?? 1} (seen:${n.links_considered ?? 0})`;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${n.node_id}</td>
        <td><span class="${pill}">${st}</span></td>
        <td>${err}</td>
        <td>${age}</td>
        <td>${sl}</td>
        <td class="small">${n.reason || ''}</td>
      `;
      tb.appendChild(tr);
    });
  }

  function applyDebugVisibility(){
    const on = document.getElementById('debugToggle').checked;
    document.querySelectorAll('.debugOnly').forEach(el=>{
      el.classList.toggle('hidden', !on);
    });
  }

  let meshChart, pairChart, stepChart, thetaChart, rttChart, sigmaBar;
  let legacyOffsetChart, legacyErrChart;
  let deltaAppliedChart, dtChart, nisChart;

  function initCharts(){
    meshChart = mkLine(document.getElementById('meshChart').getContext('2d'), 'ms');
    pairChart = mkLine(document.getElementById('pairChart').getContext('2d'), 'ms');
    stepChart = mkLine(document.getElementById('stepChart').getContext('2d'), 'ms');

    sigmaBar  = mkBar(document.getElementById('sigmaBar').getContext('2d'), 'σ latest', 'ms');
    thetaChart= mkLine(document.getElementById('thetaChart').getContext('2d'), 'ms');
    rttChart  = mkLine(document.getElementById('rttChart').getContext('2d'), 'ms');

    legacyOffsetChart = mkLine(document.getElementById('legacyOffset').getContext('2d'), 'ms');
    legacyErrChart    = mkLine(document.getElementById('legacyErr').getContext('2d'), 'ms');

    deltaAppliedChart = mkLine(document.getElementById('deltaApplied').getContext('2d'), 'ms');
    dtChart           = mkLine(document.getElementById('dtChart').getContext('2d'), 's');
    nisChart          = mkLine(document.getElementById('nisChart').getContext('2d'), 'NIS (unitless)');
  }

  function updateControllerCharts(ctrl, selectedNode){
    const byNode = ctrl || {};
    const pts = byNode[selectedNode] || [];
    const applied = pts
      .filter(p => p && p.delta_applied_ms != null && Number.isFinite(p.delta_applied_ms))
      .map(p => ({x:new Date(p.t_wall*1000), y:p.delta_applied_ms}));
    const dt = pts
      .filter(p => p && p.dt_s != null && Number.isFinite(p.dt_s))
      .map(p => ({x:new Date(p.t_wall*1000), y:p.dt_s}));

    deltaAppliedChart.data.datasets = [{label:selectedNode, data:applied, borderColor:'#2ecc71', borderWidth:1.6}];
    dtChart.data.datasets = [{label:selectedNode, data:dt, borderColor:'#3498db', borderWidth:1.6}];
    deltaAppliedChart.update(); dtChart.update();
  }

  function updateKalmanCharts(kal, selectedNode){
    const byNode = kal || {};
    const pts = byNode[selectedNode] || [];
    const nis = pts
      .filter(p => p && p.nis_med != null && Number.isFinite(p.nis_med))
      .map(p => ({x:new Date(p.t_wall*1000), y:p.nis_med}));
    nisChart.data.datasets = [{label:selectedNode, data:nis, borderColor:'#f1c40f', borderWidth:1.6}];
    nisChart.update();
  }

  async function refresh(){
    try{
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
        `now=${M.now_utc || 'n/a'} · conv_window=${M.conv_window_s || '?'}s · sources=${JSON.stringify(M.sources || {})}`
      );

      const T = (M.thresholds || {});
      document.getElementById('thrLine').textContent =
        `fresh≤${T.fresh_s ?? '?'}s · linkσmed≤${T.link_sigma_med_ms ?? '?'}ms · warmup≥${T.warmup_min_samples ?? '?'} samples · K_stable_links=${T.k_stable_links ?? '?'}`;

      renderStatusTable(ov.nodes || []);

      // mesh plots
      const style = document.getElementById('meshStyle').value || 'raw';
      const meshSeries = (style === 'smooth') ? (mesh.mesh_smooth || {}) : (mesh.mesh_raw || {});
      const stepSeries = (style === 'smooth') ? (mesh.step_smooth || {}) : (mesh.step_raw || {});
      updateLine(meshChart, meshSeries, "mesh_err_ms", "Node ");
      updateLine(stepChart, stepSeries, "step_ms", "Node ");
      updatePairs(pairChart, mesh.pairs_raw || {});

      // legacy (optional)
      const L = mesh.legacy || {};
      updateLine(legacyOffsetChart, (L.offset_ms || {}), "offset_ms", "Node ");
      updateLine(legacyErrChart, (L.err_ms || {}), "err_ms", "Node ");

      // link plots
      updateSigmaBar(sigmaBar, link.latest_sigma || {});
      updateLine(thetaChart, link.links || {}, "theta_ms", "");
      updateLine(rttChart, link.links || {}, "rtt_ms", "");

      // controller dropdown + debug
      const ctrlMap = ctrl.controller || {};
      const sel = document.getElementById('ctrlNode');
      const nodeIds = Object.keys(ctrlMap).sort();

      if(sel.options.length === 0){
        nodeIds.forEach(n=>{
          const o = document.createElement('option');
          o.value = n; o.textContent = n;
          sel.appendChild(o);
        });
        if(nodeIds.length) sel.value = nodeIds[0];
      }

      const chosen = sel.value || (nodeIds[0] || '');
      if(chosen){
        updateControllerCharts(ctrlMap, chosen);
        updateKalmanCharts(ctrl.kalman || {}, chosen);
      }

    } catch(e){
      console.error('refresh failed:', e);
    }
  }

  window.addEventListener('load', ()=>{
    initCharts();
    document.getElementById('debugToggle').addEventListener('change', applyDebugVisibility);
    document.getElementById('ctrlNode').addEventListener('change', ()=>refresh());
    document.getElementById('meshStyle').addEventListener('change', ()=>refresh());
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
    print("Starting MeshTime Dashboard v2 on http://0.0.0.0:5000/")
    app.run(host="0.0.0.0", port=5000, debug=False)
