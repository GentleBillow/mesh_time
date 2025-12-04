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

    MVP: einfach Payload loggen. Später auf C -> SQLite.
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
