# mesh/sync.py

import time
import random
from typing import List


class SyncModule:
    """
    Stub for the time sync module.
    For now:
      - mesh_time() = time.monotonic() + constant offset per node
      - send_beacons() does nothing

    Later:
      - implement 4TS exchange, θ_ij, σ_ij, inverse-variance fusion, etc.
    """

    def __init__(self, node_id: str, neighbors: List[str]):
        self.node_id = node_id
        self.neighbors = neighbors

        # For now: give each node a different offset so we can see
        # convergence later. Right now, it's static fake offset.
        self._offset = random.uniform(-0.2, 0.2)  # ±200 ms

    def mesh_time(self) -> float:
        """
        Return the node's current mesh time.
        For now: monotonic + fixed fake offset.
        """
        return time.monotonic() + self._offset

    async def send_beacons(self):
        """
        Placeholder: will send CoAP sync beacons to neighbors.
        For now: just a no-op.
        """
        # Later: build and send /sync/beacon POST to each neighbor
        return

    def inject_disturbance(self, delta: float):
        """
        For the chaos button: perturb the current offset.
        """
        self._offset += delta

    def status_snapshot(self) -> dict:
        """
        Minimal 'status' for /status endpoint and debug.
        """
        return {
            "node_id": self.node_id,
            "offset_estimate": self._offset,
            "neighbors": self.neighbors,
        }
