# mesh/node.py

import time

class MeshNode:
    def __init__(self, node_id: str, node_cfg: dict, global_cfg: dict):
        self.id = node_id
        self.cfg = node_cfg
        self.global_cfg = global_cfg

    def run(self):
        """
        For now: just a dumb loop that prints a heartbeat with the node id.
        Later we replace this with asyncio (sync_loop, sensor_loop, etc.).
        """
        print(f"[{self.id}] MeshNode starting with config: {self.cfg}")
        try:
            while True:
                print(f"[{self.id}] heartbeat...")
                time.sleep(2.0)
        except KeyboardInterrupt:
            print(f"[{self.id}] shutting down.")
