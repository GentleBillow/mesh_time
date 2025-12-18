# -*- coding: utf-8 -*-
# mesh/coap_endpoints.py
"""
CoAP endpoints — Sync + Telemetry ingest (sink-side throttling)

Drop-in replacement.

Config keys (global "sync" block in nodes.json):
  ingest_ntp_every: int (default 3)
  ingest_mesh_clock_every: int (default 2)
  ingest_link_every: int (default 8)
  ingest_obs_link_every: int (default 8)
  ingest_diag_controller_every: int (default 5)
  ingest_diag_kalman_every: int (default 8)
  ingest_sensor_every: int (default 3)

  ingest_log_ntp / ingest_log_link / ingest_log_sensor: bool (default False)
  coap_debug: bool (default False)  # adds debug fields to /sync/beacon response
"""

from __future__ import annotations

import json
import time
import logging
from typing import Any, Dict, Optional

import aiocoap
import aiocoap.resource as resource
from aiocoap.numbers.codes import Code

log = logging.getLogger("meshtime.coap")

JSON_CF = aiocoap.numbers.media_types_rev.get("application/json", 0)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _safe_json(payload: bytes) -> Dict[str, Any]:
    if not payload:
        return {}
    try:
        obj = json.loads(payload.decode("utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}

def _json_response(obj: Dict[str, Any], code=aiocoap.CONTENT) -> aiocoap.Message:
    return aiocoap.Message(
        code=code,
        payload=json.dumps(obj).encode("utf-8"),
        content_format=JSON_CF,
    )

def _boot_epoch_now() -> float:
    return time.time() - time.monotonic()

def _sync_cfg(node) -> Dict[str, Any]:
    try:
        return (getattr(node, "global_cfg", {}) or {}).get("sync", {}) or {}
    except Exception:
        return {}

def _cfg_int(node, key: str, default: int) -> int:
    cfg = _sync_cfg(node)
    try:
        v = int(cfg.get(key, default))
        return v if v > 0 else 1
    except Exception:
        return max(int(default), 1)

def _cfg_bool(node, key: str, default: bool = False) -> bool:
    cfg = _sync_cfg(node)
    v = cfg.get(key, default)
    return bool(v)

def _tick(self_obj, name: str) -> int:
    c = getattr(self_obj, name, 0) + 1
    setattr(self_obj, name, c)
    return c

def _every(counter: int, n: int) -> bool:
    if n <= 1:
        return True
    return (counter % n) == 0


# ---------------------------------------------------------------------
# Sync endpoints
# ---------------------------------------------------------------------

class SyncBeaconResource(resource.Resource):
    """
    POST /sync/beacon
    Secure via node.sync.verify_beacon(data)
    """

    def __init__(self, node):
        super().__init__()
        self.node = node
        self._debug = _cfg_bool(node, "coap_debug", False)

    async def render_post(self, request):
        t2 = time.monotonic()
        data = _safe_json(request.payload)

        if not self.node.sync.verify_beacon(data):
            return aiocoap.Message(code=Code.UNAUTHORIZED)

        src = data.get("src")
        dst = data.get("dst", self.node.id)
        dst_ok = (dst == self.node.id)

        sender_offset = data.get("offset", None)
        sender_boot_epoch = data.get("boot_epoch", None)
        sender_t1 = data.get("t1", None)

        t3 = time.monotonic()

        resp: Dict[str, Any] = {
            "t2": t2,
            "t3": t3,
            "boot_epoch": float(getattr(self.node.sync, "_boot_epoch", _boot_epoch_now())),
            "offset": float(self.node.sync.get_offset()),
        }

        if self._debug:
            resp.update(
                {
                    "receiver_id": self.node.id,
                    "dst_ok": bool(dst_ok),
                    "src": src,
                    "dst": dst,
                    "sender_offset": sender_offset,
                    "sender_boot_epoch": sender_boot_epoch,
                    "sender_t1": sender_t1,
                }
            )

        return _json_response(resp, code=aiocoap.CONTENT)


class DisturbResource(resource.Resource):
    """
    POST /sync/disturb
    Secure via node.sync.verify_disturb(data)
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_post(self, request):
        data = _safe_json(request.payload)

        if not self.node.sync.verify_disturb(data):
            return aiocoap.Message(code=Code.UNAUTHORIZED)

        try:
            delta = float(data.get("delta", 0.0))
        except Exception:
            delta = 0.0

        if delta:
            self.node.sync.inject_disturbance(delta)

        return aiocoap.Message(code=aiocoap.CHANGED)


class StatusResource(resource.Resource):
    """
    GET /status
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_get(self, request):
        snapshot = self.node.sync.status_snapshot()
        return _json_response(snapshot, code=aiocoap.CONTENT)


# ---------------------------------------------------------------------
# Relay ingest endpoints (sink node: C writes to SQLite)
# ---------------------------------------------------------------------

class RelayIngestSensorResource(resource.Resource):
    """
    POST /relay/ingest/sensor
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_post(self, request):
        data = _safe_json(request.payload)
        st = getattr(self.node, "storage", None)
        if st is None:
            return aiocoap.Message(code=aiocoap.CHANGED)

        n = _cfg_int(self.node, "ingest_sensor_every", 3)
        c = _tick(self, "_sensor_counter")
        do_write = _every(c, n)

        if _cfg_bool(self.node, "ingest_log_sensor", False) and do_write:
            log.info("[%s] ingest/sensor %s", self.node.id, data)

        if do_write:
            try:
                node_id = str(data.get("node_id", "unknown"))
                t_mesh = float(data.get("t_mesh", 0.0))
                monotonic = data.get("monotonic", None)
                monotonic_f: Optional[float] = float(monotonic) if monotonic is not None else None
                value = data.get("value", None)
                value_f = float(value) if value is not None else 0.0
                raw_json = json.dumps(data)

                st.insert_reading(
                    node_id=node_id,
                    t_mesh=t_mesh,
                    value=value_f,
                    monotonic=monotonic_f,
                    raw_json=raw_json,
                )
            except Exception as e:
                log.warning("[%s] ingest/sensor DB insert failed: %s", self.node.id, type(e).__name__)

        return aiocoap.Message(code=aiocoap.CHANGED)


class RelayIngestNtpResource(resource.Resource):
    """
    POST /relay/ingest/ntp
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_post(self, request):
        data = _safe_json(request.payload)
        st = getattr(self.node, "storage", None)
        if st is None:
            return aiocoap.Message(code=aiocoap.CHANGED)

        n_ntp = _cfg_int(self.node, "ingest_ntp_every", 3)
        n_mesh_clock = _cfg_int(self.node, "ingest_mesh_clock_every", 2)
        n_diag_ctrl = _cfg_int(self.node, "ingest_diag_controller_every", 5)

        c = _tick(self, "_ntp_counter")
        do_ntp = _every(c, n_ntp)
        do_mesh_clock = _every(c, n_mesh_clock)
        do_diag_ctrl = _every(c, n_diag_ctrl)

        if not do_ntp:
            return aiocoap.Message(code=aiocoap.CHANGED)

        if _cfg_bool(self.node, "ingest_log_ntp", False):
            log.info("[%s] ingest/ntp %s", self.node.id, data)

        try:
            node_id = str(data.get("node_id", "unknown"))
            t_wall = float(data.get("t_wall", time.time()))
            t_mono = float(data.get("t_mono", 0.0))
            t_mesh = float(data.get("t_mesh", 0.0))
            offset = float(data.get("offset", 0.0))

            err = data.get("err_mesh_vs_wall", None)
            if err is None:
                err = t_mesh - t_wall
            err = float(err)

            # Controller debug (optional)
            delta_desired_ms = data.get("delta_desired_ms", None)
            delta_applied_ms = data.get("delta_applied_ms", None)
            dt_s = data.get("dt_s", None)
            slew_clipped = data.get("slew_clipped", None)

            delta_desired_ms = float(delta_desired_ms) if delta_desired_ms is not None else None
            delta_applied_ms = float(delta_applied_ms) if delta_applied_ms is not None else None
            dt_s = float(dt_s) if dt_s is not None else None
            slew_clipped_b = None if slew_clipped is None else bool(slew_clipped)

            # 1) ntp_reference (node snapshot)
            st.insert_ntp_reference(
                node_id=node_id,
                t_wall=t_wall,
                t_mono=t_mono,
                t_mesh=t_mesh,
                offset=offset,
                err_mesh_vs_wall=err,
                peer_id=None,
                theta_ms=None,
                rtt_ms=None,
                sigma_ms=None,
                delta_desired_ms=delta_desired_ms,
                delta_applied_ms=delta_applied_ms,
                dt_s=dt_s,
                slew_clipped=slew_clipped_b,
            )

            # 2) mesh_clock (used by /api/sync) — downsampled separately
            if do_mesh_clock:
                try:
                    st.insert_mesh_clock(
                        node_id=node_id,
                        t_wall_s=t_wall,
                        t_mono_s=t_mono,
                        t_mesh_s=t_mesh,
                        offset_s=offset,
                        err_mesh_vs_wall_s=err,
                    )
                except Exception:
                    pass

            # 3) diag_controller (rare)
            if do_diag_ctrl and ((dt_s is not None) or (delta_desired_ms is not None) or (delta_applied_ms is not None)):
                try:
                    st.insert_diag_controller(
                        node_id=node_id,
                        dt_s=dt_s,
                        delta_desired_ms=delta_desired_ms,
                        delta_applied_ms=delta_applied_ms,
                        slew_clipped=slew_clipped_b,
                        max_slew_ms_s=None,
                        eff_eta=None,
                    )
                except Exception:
                    pass

        except Exception as e:
            log.warning("[%s] ingest/ntp DB insert failed: %s", self.node.id, type(e).__name__)

        return aiocoap.Message(code=aiocoap.CHANGED)


class RelayIngestLinkResource(resource.Resource):
    """
    POST /relay/ingest/link
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_post(self, request):
        data = _safe_json(request.payload)
        st = getattr(self.node, "storage", None)
        if st is None:
            return aiocoap.Message(code=aiocoap.CHANGED)

        n_link = _cfg_int(self.node, "ingest_link_every", 8)
        n_obs = _cfg_int(self.node, "ingest_obs_link_every", 8)

        c = _tick(self, "_link_counter")
        do_link = _every(c, n_link)
        do_obs = _every(c, n_obs)

        if not do_link:
            return aiocoap.Message(code=aiocoap.CHANGED)

        if _cfg_bool(self.node, "ingest_log_link", False):
            log.info("[%s] ingest/link %s", self.node.id, data)

        try:
            node_id = str(data.get("node_id", "unknown"))
            peer_id = data.get("peer_id", None)
            if peer_id is None:
                return aiocoap.Message(code=aiocoap.CHANGED)
            peer_id = str(peer_id)

            theta_ms = data.get("theta_ms", None)
            rtt_ms = data.get("rtt_ms", None)
            sigma_ms = data.get("sigma_ms", None)

            theta_ms = float(theta_ms) if theta_ms is not None else None
            rtt_ms = float(rtt_ms) if rtt_ms is not None else None
            sigma_ms = float(sigma_ms) if sigma_ms is not None else None

            t_wall = float(data.get("t_wall", time.time()))
            t_mono = float(data.get("t_mono", 0.0))
            t_mesh = float(data.get("t_mesh", 0.0))
            offset = float(data.get("offset", 0.0))
            err = t_mesh - t_wall

            # Optional controller debug
            delta_desired_ms = data.get("delta_desired_ms", None)
            delta_applied_ms = data.get("delta_applied_ms", None)
            dt_s = data.get("dt_s", None)
            slew_clipped = data.get("slew_clipped", None)

            delta_desired_ms = float(delta_desired_ms) if delta_desired_ms is not None else None
            delta_applied_ms = float(delta_applied_ms) if delta_applied_ms is not None else None
            dt_s = float(dt_s) if dt_s is not None else None
            slew_clipped_b = None if slew_clipped is None else bool(slew_clipped)

            # ntp_reference (link-specific fields)
            st.insert_ntp_reference(
                node_id=node_id,
                t_wall=t_wall,
                t_mono=t_mono,
                t_mesh=t_mesh,
                offset=offset,
                err_mesh_vs_wall=float(err),
                peer_id=peer_id,
                theta_ms=theta_ms,
                rtt_ms=rtt_ms,
                sigma_ms=sigma_ms,
                delta_desired_ms=delta_desired_ms,
                delta_applied_ms=delta_applied_ms,
                dt_s=dt_s,
                slew_clipped=slew_clipped_b,
            )

            # obs_link (raw link measurement) — also downsampled
            if do_obs:
                try:
                    st.insert_obs_link(
                        node_id=node_id,
                        peer_id=peer_id,
                        theta_ms=theta_ms,
                        rtt_ms=rtt_ms,
                        sigma_ms=sigma_ms,
                        t1_s=float(data["t1_s"]) if data.get("t1_s") is not None else None,
                        t2_s=float(data["t2_s"]) if data.get("t2_s") is not None else None,
                        t3_s=float(data["t3_s"]) if data.get("t3_s") is not None else None,
                        t4_s=float(data["t4_s"]) if data.get("t4_s") is not None else None,
                        accepted=True,
                        weight=float(data["weight"]) if data.get("weight") is not None else None,
                        reject_reason=None,
                    )
                except Exception as e:
                    log.warning("[%s] ingest/link obs_link insert failed: %s", self.node.id, type(e).__name__)

        except Exception as e:
            log.warning("[%s] ingest/link DB insert failed: %s", self.node.id, type(e).__name__)

        return aiocoap.Message(code=aiocoap.CHANGED)


class RelayIngestMeshClockResource(resource.Resource):
    """
    POST /relay/ingest/mesh_clock
    NOTE: If you already write mesh_clock via ingest/ntp, you can throttle this aggressively.
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_post(self, request):
        data = _safe_json(request.payload)
        st = getattr(self.node, "storage", None)
        if st is None:
            return aiocoap.Message(code=aiocoap.CHANGED)

        n = _cfg_int(self.node, "ingest_mesh_clock_every", 2)
        c = _tick(self, "_mesh_clock_counter")
        if not _every(c, n):
            return aiocoap.Message(code=aiocoap.CHANGED)

        try:
            node_id = str(data.get("node_id", "unknown"))
            t_wall_s = float(data.get("t_wall_s", time.time()))
            t_mono_s = float(data.get("t_mono_s", 0.0))
            t_mesh_s = float(data.get("t_mesh_s", 0.0))
            offset_s = float(data.get("offset_s", 0.0))
            err = data.get("err_mesh_vs_wall_s", None)
            if err is None:
                err = t_mesh_s - t_wall_s
            err = float(err)

            st.insert_mesh_clock(
                node_id=node_id,
                t_wall_s=t_wall_s,
                t_mono_s=t_mono_s,
                t_mesh_s=t_mesh_s,
                offset_s=offset_s,
                err_mesh_vs_wall_s=err,
            )
        except Exception as e:
            log.warning("[%s] ingest/mesh_clock DB insert failed: %s", self.node.id, type(e).__name__)

        return aiocoap.Message(code=aiocoap.CHANGED)


class RelayIngestDiagControllerResource(resource.Resource):
    """
    POST /relay/ingest/diag_controller
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_post(self, request):
        data = _safe_json(request.payload)
        st = getattr(self.node, "storage", None)
        if st is None:
            return aiocoap.Message(code=aiocoap.CHANGED)

        n = _cfg_int(self.node, "ingest_diag_controller_every", 5)
        c = _tick(self, "_diag_ctrl_counter")
        if not _every(c, n):
            return aiocoap.Message(code=aiocoap.CHANGED)

        try:
            node_id = str(data.get("node_id", "unknown"))
            dt_s = data.get("dt_s", None)
            delta_desired_ms = data.get("delta_desired_ms", None)
            delta_applied_ms = data.get("delta_applied_ms", None)
            slew_clipped = data.get("slew_clipped", None)
            max_slew_ms_s = data.get("max_slew_ms_s", None)
            eff_eta = data.get("eff_eta", None)

            dt_s = float(dt_s) if dt_s is not None else None
            delta_desired_ms = float(delta_desired_ms) if delta_desired_ms is not None else None
            delta_applied_ms = float(delta_applied_ms) if delta_applied_ms is not None else None
            max_slew_ms_s = float(max_slew_ms_s) if max_slew_ms_s is not None else None
            eff_eta = float(eff_eta) if eff_eta is not None else None
            slew_clipped_b = None if slew_clipped is None else bool(slew_clipped)

            st.insert_diag_controller(
                node_id=node_id,
                dt_s=dt_s,
                delta_desired_ms=delta_desired_ms,
                delta_applied_ms=delta_applied_ms,
                slew_clipped=slew_clipped_b,
                max_slew_ms_s=max_slew_ms_s,
                eff_eta=eff_eta,
            )
        except Exception as e:
            log.warning("[%s] ingest/diag_controller DB insert failed: %s", self.node.id, type(e).__name__)

        return aiocoap.Message(code=aiocoap.CHANGED)


class RelayIngestDiagKalmanResource(resource.Resource):
    """
    POST /relay/ingest/diag_kalman
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_post(self, request):
        data = _safe_json(request.payload)
        st = getattr(self.node, "storage", None)
        if st is None:
            return aiocoap.Message(code=aiocoap.CHANGED)

        n = _cfg_int(self.node, "ingest_diag_kalman_every", 8)
        c = _tick(self, "_diag_kalman_counter")
        if not _every(c, n):
            return aiocoap.Message(code=aiocoap.CHANGED)

        try:
            node_id = str(data.get("node_id", "unknown"))
            n_meas = int(data.get("n_meas", 0))

            def f(key):
                v = data.get(key, None)
                return float(v) if v is not None else None

            st.insert_diag_kalman(
                node_id=node_id,
                n_meas=n_meas,
                innov_med_ms=f("innov_med_ms"),
                innov_p95_ms=f("innov_p95_ms"),
                nis_med=f("nis_med"),
                nis_p95=f("nis_p95"),
                x_offset_ms=f("x_offset_ms"),
                x_drift_ppm=f("x_drift_ppm"),
                p_offset_ms2=f("p_offset_ms2"),
                p_drift_ppm2=f("p_drift_ppm2"),
                r_eff_ms2=f("r_eff_ms2"),
            )
        except Exception as e:
            log.warning("[%s] ingest/diag_kalman DB insert failed: %s", self.node.id, type(e).__name__)

        return aiocoap.Message(code=aiocoap.CHANGED)


# ---------------------------------------------------------------------
# Build site
# ---------------------------------------------------------------------

def build_site(node) -> resource.Site:
    root = resource.Site()

    root.add_resource(("sync", "beacon"), SyncBeaconResource(node))
    root.add_resource(("sync", "disturb"), DisturbResource(node))

    root.add_resource(("status",), StatusResource(node))

    root.add_resource(("relay", "ingest", "sensor"), RelayIngestSensorResource(node))
    root.add_resource(("relay", "ingest", "ntp"), RelayIngestNtpResource(node))
    root.add_resource(("relay", "ingest", "link"), RelayIngestLinkResource(node))

    root.add_resource(("relay", "ingest", "mesh_clock"), RelayIngestMeshClockResource(node))
    root.add_resource(("relay", "ingest", "diag_controller"), RelayIngestDiagControllerResource(node))
    root.add_resource(("relay", "ingest", "diag_kalman"), RelayIngestDiagKalmanResource(node))

    return root
