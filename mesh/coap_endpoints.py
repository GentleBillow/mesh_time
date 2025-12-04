# mesh/coap_endpoints.py

import json
import time

import aiocoap
import aiocoap.resource as resource


class SyncBeaconResource(resource.Resource):
    """
    POST /sync/beacon

    Request payload (JSON):
      { "src": "A", "dst": "B", "t1": <float> }

    Response payload (JSON):
      { "src": "B", "dst": "A", "t2": <float>, "t3": <float> }

    All times are local monotonic seconds (time.monotonic()).
    """

    def __init__(self, node):
        super(SyncBeaconResource, self).__init__()
        self.node = node

    async def render_post(self, request):
        t2 = time.monotonic()

        try:
            data = json.loads(request.payload.decode("utf-8"))
        except Exception:
            data = {}

        src = data.get("src", None)
        dst = data.get("dst", self.node.id)

        # t3 = time we send the reply (close to t2, one extra monotonic call)
        t3 = time.monotonic()

        resp_payload = {
            "src": self.node.id,
            "dst": src or "unknown",
            "t2": t2,
            "t3": t3,
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

    If no payload or invalid, does nothing.
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

    Returns JSON with current offset, neighbors, etc.
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

    MVP: just print payload. Later, on node C, this will write into SQLite.
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
