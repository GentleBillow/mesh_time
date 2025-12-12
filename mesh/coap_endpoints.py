# -*- coding: utf-8 -*-
# mesh/coap_endpoints.py
"""
CoAP endpoints — drop-in, aufgeräumt

Wichtigste Fixes:
- /sync/beacon: validiert dst, antwortet immer sauber (auch bei kaputtem JSON)
- boot_epoch IMMER mitliefern (für monotonic->epoch lifting)
- Keine Debug-Felder im Default-Antwortpayload (nur optional via cfg), damit Payload klein bleibt
- /relay/ingest/*: robust gegen fehlende Felder, keine Exceptions -> Server bleibt stabil
- content_format korrekt gesetzt (application/json)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import aiocoap
import aiocoap.resource as resource


JSON_CF = aiocoap.numbers.media_types_rev.get("application/json", 0)


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

        # Optional: debug payload size toggle
        sync_cfg = {}
        try:
            sync_cfg = (getattr(node, "global_cfg", {}) or {}).get("sync", {}) or {}
        except Exception:
            sync_cfg = {}
        self._debug = bool(sync_cfg.get("coap_debug", False))

    async def render_post(self, request):
        t2 = time.monotonic()
        data = _safe_json(request.payload)

        src = data.get("src")
        dst = data.get("dst", self.node.id)

        # If someone misroutes, still respond but mark it
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
      { "delta": <float seconds> }
    """

    def __init__(self, node):
        super().__init__()
        self.node = node

    async def render_post(self, request):
        data = _safe_json(request.payload)
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
        # print is okay for now; later: rate-limit / structured logging
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

    async def render_post(self, request):
        data = _safe_json(request.payload)
        print(f"[{self.node.id}] relay/ingest/ntp: {data}")

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

                st.insert_ntp_reference(
                    node_id=node_id,
                    t_wall=t_wall,
                    t_mono=t_mono,
                    t_mesh=t_mesh,
                    offset=offset,
                    err_mesh_vs_wall=err,
                )
            except Exception as e:
                print(f"[{self.node.id}] relay/ingest/ntp: DB insert failed: {e}")

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

    return root
