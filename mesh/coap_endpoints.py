# -*- coding: utf-8 -*-
# mesh/coap_endpoints.py - FIXED VERSION mit Link-Metrics Endpoint (+ Link-Logging Throttle)
"""
CoAP endpoints — mit /relay/ingest/link für Link-Metriken

FIXES:
- Bugfix: _ntp_counter korrekt initialisiert (kein AttributeError)
- Link-Logging gedrosselt: obs_link wird nur jedes N-te Paket gespeichert
- CoAP handler geben immer eine Response zurück
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import aiocoap
import aiocoap.resource as resource
from aiocoap.numbers.codes import Code


JSON_CF = aiocoap.numbers.media_types_rev.get("application/json", 0)

# ---------------------------------------------------------------------------
# Quick-fix knobs (IO/SD-card saver)
# ---------------------------------------------------------------------------

# Speichere obs_link (teuer!) nur jedes N-te Link-Paket
LINK_LOG_EVERY_N = 3  # 2 = halbieren, 3 = dritteln, 4 = vierteln ...


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
    # epoch at monotonic=0
    return time.time() - time.monotonic()


class SyncBeaconResource(resource.Resource):
    """
    POST /sync/beacon

    Request JSON:
      {
        "src": "<sender_id>",
        "dst": "<receiver_id>",
        "t1":  <sender_monotonic>,
        "boot_epoch": <sender_boot_epoch>,     # strongly recommended
        "offset": <sender_offset_seconds>
      }

    Response JSON:
      {
        "t2": <receiver_monotonic_at_receive>,
        "t3": <receiver_monotonic_before_reply>,
        "boot_epoch": <receiver_boot_epoch>,
        "offset": <receiver_offset_seconds>
      }
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

        sync_cfg = {}
        try:
            sync_cfg = (getattr(node, "global_cfg", {}) or {}).get("sync", {}) or {}
        except Exception:
            sync_cfg = {}
        self._debug = bool(sync_cfg.get("coap_debug", False))

    async def render_post(self, request):
        t2 = time.monotonic()
        data = _safe_json(request.payload)

        # --- secure measurement endpoint (optional via PSK) ---
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
            "boot_epoch": _boot_epoch_now(),
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

    JSON:
      { "src": "<sender_id>", "dst": "<receiver_id>", "delta": <float seconds>, "auth": "<hmac>" }
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


class RelayIngestSensorResource(resource.Resource):
    """
    POST /relay/ingest/sensor

    Example JSON:
      {
        "node_id": "B",
        "t_mesh": 1234.567,
        "monotonic": 1230.123,
        "value": 42.0,
        "extra": {...}
      }
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_post(self, request):
        data = _safe_json(request.payload)
        # Achtung: print = IO. Für Quickfix ok, später ggf. drosseln.
        print(f"[{self.node.id}] relay/ingest/sensor: {data}")

        st = getattr(self.node, "storage", None)
        if st is not None:
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
                print(f"[{self.node.id}] relay/ingest/sensor: DB insert failed: {e}")

        return aiocoap.Message(code=aiocoap.CHANGED)


class RelayIngestNtpResource(resource.Resource):
    """
    POST /relay/ingest/ntp

    JSON:
      {
        "node_id": "B",
        "t_wall":  <float epoch seconds>,
        "t_mono":  <float monotonic seconds>,
        "t_mesh":  <float mesh time seconds>,
        "offset":  <float seconds>,
        "err_mesh_vs_wall": <float seconds>   # optional
      }
    """

    def __init__(self, node):
        super().__init__()
        self.node = node
        self._ntp_counter = 0  # FIX: korrekt initialisieren

    async def render_post(self, request):
        data = _safe_json(request.payload)

        # NTP NICHT drosseln (du wolltest Link drosseln) – aber counter bleibt als Option drin.
        self._ntp_counter += 1

        st = getattr(self.node, "storage", None)
        if st is not None:
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

                # --- controller debug fields (optional) ---
                delta_desired_ms = data.get("delta_desired_ms", None)
                delta_applied_ms = data.get("delta_applied_ms", None)
                dt_s = data.get("dt_s", None)
                slew_clipped = data.get("slew_clipped", None)

                delta_desired_ms = float(delta_desired_ms) if delta_desired_ms is not None else None
                delta_applied_ms = float(delta_applied_ms) if delta_applied_ms is not None else None
                dt_s = float(dt_s) if dt_s is not None else None
                slew_clipped_b = None if slew_clipped is None else bool(slew_clipped)

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

                # mesh_clock (optional)
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

                # diag_controller (optional)
                if (dt_s is not None) or (delta_desired_ms is not None) or (delta_applied_ms is not None):
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
                print(f"[{self.node.id}] relay/ingest/ntp: DB insert failed: {e}")

        return aiocoap.Message(code=aiocoap.CHANGED)


class RelayIngestLinkResource(resource.Resource):
    """
    POST /relay/ingest/link

    Empfängt Link-Metriken via Telemetrie von Nodes ohne Storage.

    JSON:
      {
        "node_id": "B",
        "peer_id": "C",
        "theta_ms": 1.23,
        "rtt_ms": 2.45,
        "sigma_ms": 0.12,
        "t_wall": <epoch>,
        "t_mono": <monotonic>,
        "t_mesh": <mesh_time>,
        "offset": <node_offset>,
        "t1_s": ..., "t2_s": ..., "t3_s": ..., "t4_s": ...,
        "weight": ...
      }
    """

    def __init__(self, node):
        super().__init__()
        self.node = node
        self._link_counter = 0  # FIX: eigener Counter für Link-Logging

    async def render_post(self, request):
        data = _safe_json(request.payload)

        self._link_counter += 1
        log_obs_link = (self._link_counter % int(LINK_LOG_EVERY_N) == 0)

        st = getattr(self.node, "storage", None)
        if st is not None:
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

                # controller debug fields (optional)
                delta_desired_ms = data.get("delta_desired_ms", None)
                delta_applied_ms = data.get("delta_applied_ms", None)
                dt_s = data.get("dt_s", None)
                slew_clipped = data.get("slew_clipped", None)

                delta_desired_ms = float(delta_desired_ms) if delta_desired_ms is not None else None
                delta_applied_ms = float(delta_applied_ms) if delta_applied_ms is not None else None
                dt_s = float(dt_s) if dt_s is not None else None
                slew_clipped_b = None if slew_clipped is None else bool(slew_clipped)

                # ntp_reference: kann bleiben (leichter/aggregierter als obs_link)
                st.insert_ntp_reference(
                    node_id=node_id,
                    t_wall=t_wall,
                    t_mono=t_mono,
                    t_mesh=t_mesh,
                    offset=offset,
                    err_mesh_vs_wall=err,
                    peer_id=peer_id,
                    theta_ms=theta_ms,
                    rtt_ms=rtt_ms,
                    sigma_ms=sigma_ms,
                    delta_desired_ms=delta_desired_ms,
                    delta_applied_ms=delta_applied_ms,
                    dt_s=dt_s,
                    slew_clipped=slew_clipped_b,
                )

                # obs_link: TEUER -> gedrosselt
                if log_obs_link:
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
                        print(f"[{self.node.id}] relay/ingest/link: obs_link insert failed: {e}")

            except Exception as e:
                print(f"[{self.node.id}] relay/ingest/link: DB insert failed: {e}")

        return aiocoap.Message(code=aiocoap.CHANGED)


class RelayIngestMeshClockResource(resource.Resource):
    """
    POST /relay/ingest/mesh_clock
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_post(self, request):
        data = _safe_json(request.payload)
        st = getattr(self.node, "storage", None)
        if st is not None:
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
                print(f"[{self.node.id}] relay/ingest/mesh_clock: DB insert failed: {e}")
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
        if st is not None:
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
                print(f"[{self.node.id}] relay/ingest/diag_controller: DB insert failed: {e}")
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
        if st is not None:
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
                print(f"[{self.node.id}] relay/ingest/diag_kalman: DB insert failed: {e}")
        return aiocoap.Message(code=aiocoap.CHANGED)


def build_site(node) -> resource.Site:
    """
    Build the CoAP resource tree for a node.
    """
    root = resource.Site()

    # Sync endpoints
    root.add_resource(("sync", "beacon"), SyncBeaconResource(node))
    root.add_resource(("sync", "disturb"), DisturbResource(node))

    # Status
    root.add_resource(("status",), StatusResource(node))

    # Relay ingests
    root.add_resource(("relay", "ingest", "sensor"), RelayIngestSensorResource(node))
    root.add_resource(("relay", "ingest", "ntp"), RelayIngestNtpResource(node))
    root.add_resource(("relay", "ingest", "link"), RelayIngestLinkResource(node))

    root.add_resource(("relay", "ingest", "mesh_clock"), RelayIngestMeshClockResource(node))
    root.add_resource(("relay", "ingest", "diag_controller"), RelayIngestDiagControllerResource(node))
    root.add_resource(("relay", "ingest", "diag_kalman"), RelayIngestDiagKalmanResource(node))

    return root
