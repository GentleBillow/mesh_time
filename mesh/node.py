# mesh/node.py

import asyncio
from typing import Dict, Any

from .sync import SyncModule
from .sensor import DummySensor
from .led import DummyLED


class MeshNode:
    def __init__(self, node_id: str, node_cfg: Dict[str, Any], global_cfg: Dict[str, Any]):
        self.id = node_id
        self.cfg = node_cfg
        self.global_cfg = global_cfg

        neighbors = node_cfg.get("neighbors", [])
        self.sync = SyncModule(node_id=node_id, neighbors=neighbors)

        # For now, use dummy sensor & LED; later we swap to Grove implementations.
        self.sensor = DummySensor(sensor_type=node_cfg.get("sensor_type", "dummy"))
        self.led = DummyLED(pin=node_cfg.get("led_pin", 18))

        # placeholder for parent; used later by routing
        self.parent = node_cfg.get("parent")

        # disturbance: which node has a button; D will use this
        self.button_pin = node_cfg.get("button_pin")

        # stop flag if ever needed
        self._stop = asyncio.Event()

    async def sync_loop(self):
        """
        Periodically send sync beacons (later).
        For now: just wait with jitter and do nothing.
        """
        print(f"[{self.id}] sync_loop started")
        while not self._stop.is_set():
            await self.sync.send_beacons()
            # jittered interval ~1s, but we don't do real work yet
            await asyncio.sleep(1.0)

    async def sensor_loop(self):
        """
        Sample the sensor periodically and (later) forward readings.
        For now, just print them with mesh timestamps.
        """
        print(f"[{self.id}] sensor_loop started")
        while not self._stop.is_set():
            value = self.sensor.read()
            t_mesh = self.sync.mesh_time()
            print(f"[{self.id}] sensor reading: value={value:.3f}, t_mesh={t_mesh:.3f}")
            await asyncio.sleep(1.0)

    async def led_loop(self):
        """
        Update LED blink based on mesh time each 10 ms.
        """
        print(f"[{self.id}] led_loop started")
        while not self._stop.is_set():
            t_mesh = self.sync.mesh_time()
            self.led.update(t_mesh)
            await asyncio.sleep(0.01)

    async def run_async(self):
        """
        Start all node tasks. CoAP server will join here later.
        """
        print(f"[{self.id}] MeshNode starting with cfg: {self.cfg}")
        await asyncio.gather(
            self.sync_loop(),
            self.sensor_loop(),
            self.led_loop(),
        )

    def run(self):
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print(f"[{self.id}] shutting down (KeyboardInterrupt)")
