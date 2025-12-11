# -*- coding: utf-8 -*-
# mesh/coap_endpoints.py

import json
import time

import aiocoap
import aiocoap.resource as resource


class SyncBeaconResource(resource.Resource):
    """
    POST /sync/beacon

    Request payload (JSON):
      {
        "src": "<sender_node_id>",
        "dst": "<receiver_node_id>",
        "t1":  <sender_monotonic>,
        "offset": <sender_offset_seconds>
      }

    Response payload (JSON):
      {
        "src": "<receiver_node_id>",
        "dst": "<sender_node_id>",
        "t2":  <receiver_monotonic_before_processing>,
        "t3":  <receiver_monotonic_before_reply>,
        "offset": <receiver_offset_seconds>,

        # reine Debug-Felder:
        "receiver_id": "<receiver_node_id>",
        "sender_id":   "<sender_node_id>",
        "sender_offset": <sender_offset_seconds>
      }

    Alle Zeiten sind lokale monotonic-Sekunden (time.monotonic()).
    Offsets sind Sekunden (float).
    """

    def __init__(self, node):
        super(SyncBeaconResource, self).__init__()
        self.node = node

    async def render_post(self, request):
        # t2: Eingang
        t2 = time.monotonic()

        try:
            data = json.loads(request.payload.decode("utf-8"))
        except Exception:
            data = {}

        src = data.get("src", None)
        dst = data.get("dst", self.node.id)
        sender_offset = data.get("offset", None)

        # t3: kurz vor Antwort
        t3 = time.monotonic()

        resp_payload = {
            "src": self.node.id,
            "dst": src or "unknown",
            "t2": t2,
            "t3": t3,
            "offset": self.node.sync.get_offset(),  # <- wichtiger Teil für SyncModule

            # Bonus-Infos für Debugging
            "receiver_id": self.node.id,
            "sender_id": src,
            "sender_offset": sender_offset,
        }

        payload_bytes = json.dumps(resp_payload).encode("utf-8")
        cf = aiocoap.numbers.media_types_rev.get("application/json", 0)

        return aiocoap.Message(
            code=aiocoap.CONTENT,
            payload=payload_bytes,
            content_format=cf,
        )


class DisturbResource(resource.Resource):
    """
    POST /sync/disturb

    Payload JSON:
      { "delta": <float seconds> }

    delta wird 1:1 an SyncModule.inject_disturbance(delta) weitergegeben.
    """

    def __init__(self, node):
        super(DisturbResource, self).__init__()
        self.node = node

    async def render_post(self, request):
        try:
            data = json.loads(request.payload.decode("utf-8"))
        except Exception:
            data = {}

        delta = float(data.get("delta", 0.0))
        if delta != 0.0:
            self.node.sync.inject_disturbance(delta)

        return aiocoap.Message(code=aiocoap.CHANGED)


class StatusResource(resource.Resource):
    """
    GET /status

    Gibt den Status-Snapshot von SyncModule zurück, inkl.
    - offset_estimate / offset_estimate_ms
    - mesh_time, monotonic_now
    - peer_offsets(_ms), peer_sigma(_ms)
    - neighbors, sync_config
    """

    def __init__(self, node):
        super(StatusResource, self).__init__()
        self.node = node

    async def render_get(self, request):
        snapshot = self.node.sync.status_snapshot()
        payload_bytes = json.dumps(snapshot).encode("utf-8")
        cf = aiocoap.numbers.media_types_rev.get("application/json", 0)

        return aiocoap.Message(
            code=aiocoap.CONTENT,
            payload=payload_bytes,
            content_format=cf,
        )


class RelayIngestSensorResource(resource.Resource):
    """
    POST /relay/ingest/sensor

    Payload JSON (Beispiel):
      {
        "node_id": "B",
        "t_mesh": 1234.567,
        "monotonic": 1230.123,
        "value": 42.0,
        "extra": { ... }        # optional, wird als raw_json gespeichert
      }

    Auf dem Sammelknoten (z.B. C) wird das in SQLite geschrieben.
    """

    def __init__(self, node):
        super(RelayIngestSensorResource, self).__init__()
        self.node = node

    async def render_post(self, request):
        try:
            data = json.loads(request.payload.decode("utf-8"))
        except Exception:
            data = {}

        print("[{}] relay/ingest/sensor: {}".format(self.node.id, data))

        # Versuchen, in DB zu schreiben, falls vorhanden
        if getattr(self.node, "storage", None) is not None:
            node_id = str(data.get("node_id", "unknown"))
            t_mesh = float(data.get("t_mesh", 0.0))
            monotonic = data.get("monotonic")
            if monotonic is not None:
                monotonic = float(monotonic)

            value = data.get("value")
            if value is not None:
                value = float(value)

            # Alles übrige als raw_json speichern
            raw_json = json.dumps(data)

            try:
                self.node.storage.insert_reading(
                    node_id=node_id,
                    t_mesh=t_mesh,
                    value=value if value is not None else 0.0,
                    monotonic=monotonic,
                    raw_json=raw_json,
                )
            except Exception as e:
                print("[{}] relay/ingest/sensor: DB insert failed: {}".format(self.node.id, e))

        return aiocoap.Message(code=aiocoap.CHANGED)




def build_site(node):
    """
    Build the CoAP resource tree for a node.
    """
    root = resource.Site()

    # Sync endpoints
    root.add_resource(('sync', 'beacon'), SyncBeaconResource(node))
    root.add_resource(('sync', 'disturb'), DisturbResource(node))

    # Status
    root.add_resource(('status',), StatusResource(node))

    # Sensor relay
    root.add_resource(('relay', 'ingest', 'sensor'), RelayIngestSensorResource(node))

    return root
