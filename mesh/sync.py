# mesh/sync.py

import time
import random
import platform
from typing import List, Dict

import aiocoap
import json

IS_WINDOWS = platform.system() == "Windows"


class SyncModule:
    """
    Time sync module.

    - mesh_time() = monotonic + offset
    - offset is updated from 4-timestamp measurements to neighbors
    - On Windows, send_beacons() is a no-op (so you can dev without real peers)
    """

    def __init__(self, node_id: str, neighbors: List[str], neighbor_ips: Dict[str, str]):
        self.node_id = node_id
        self.neighbors = neighbors
        self.neighbor_ips = neighbor_ips  # e.g. {"B": "192.168.1.21", ...}

        # Start with some random offset so convergence is visible later
        self._offset = random.uniform(-0.2, 0.2)  # ±200 ms

        # Last offset per neighbor (for now, simple mean)
        self._peer_offsets: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mesh_time(self) -> float:
        """Return the node's current mesh time."""
        return time.monotonic() + self._offset

    def inject_disturbance(self, delta: float):
        """Chaos button: perturb the current offset."""
        self._offset += delta
        print(f"[{self.node_id}] inject_disturbance: delta={delta*1000:.1f} ms, "
              f"new offset={self._offset*1000:.1f} ms")

    def status_snapshot(self) -> dict:
        """For /status endpoint."""
        return {
            "node_id": self.node_id,
            "offset_estimate": self._offset,
            "peer_offsets": self._peer_offsets,
            "neighbors": self.neighbors,
        }

    async def send_beacons(self, client_ctx: aiocoap.Context):
        """
        Periodically called from MeshNode.sync_loop().

        For each neighbor:
          - send POST /sync/beacon with t1 and src id
          - receive response with t2, t3
          - measure t4 locally
          - compute θ_ij and δ_ij
          - update self._offset as average of peer offsets (simple version)

        On Windows dev machines, we skip actual network to avoid aiocoap issues.
        """
        if IS_WINDOWS:
            # Dev mode: don't actually try to talk to 192.168.x.x from laptop.
            return

        for peer in self.neighbors:
            ip = self.neighbor_ips.get(peer)
            if not ip:
                continue

            uri = f"coap://{ip}/sync/beacon"

            t1 = time.monotonic()
            payload = {
                "src": self.node_id,
                "dst": peer,
                "t1": t1,
            }

            req = aiocoap.Message(
                code=aiocoap.POST,
                uri=uri,
                payload=json.dumps(payload).encode("utf-8"),
            )

            try:
                resp = await client_ctx.request(req).response
            except Exception as e:
                print(f"[{self.node_id}] beacon to {peer} failed: {e}")
                continue

            t4 = time.monotonic()

            try:
                data = json.loads(resp.payload.decode("utf-8"))
                t2 = float(data["t2"])
                t3 = float(data["t3"])
            except Exception as e:
                print(f"[{self.node_id}] invalid beacon reply from {peer}: {e}")
                continue

            # NTP-style 4TS formulas
            rtt = (t4 - t1) - (t3 - t2)
            theta = ((t2 - t1) + (t3 - t4)) / 2.0

            self._peer_offsets[peer] = theta

            # Simple average of all peer offsets for now
            if self._peer_offsets:
                avg_theta = sum(self._peer_offsets.values()) / len(self._peer_offsets)
                self._offset = avg_theta

            print(
                f"[{self.node_id}] sync with {peer}: "
                f"theta={theta*1000:.2f} ms, rtt={rtt*1000:.2f} ms, "
                f"offset={self._offset*1000:.2f} ms"
            )
