# mesh/node.py

import asyncio
import platform
from typing import Dict, Any, Optional

import aiocoap

from .sync import SyncModule
from .sensor import DummySensor
from .led import DummyLED
from .coap_endpoints import build_site

IS_WINDOWS = platform.system() == "Windows"


class MeshNode:
    def __init__(self, node_id: str, node_cfg: Dict[str, Any], global_cfg: Dict[str, Any]):
        self.id = node_id
        self.cfg = node_cfg
        self.global_cfg = global_cfg

        # ------------------------------------------------------------------
        # Neighbors & their IPs (from global config)
        # ------------------------------------------------------------------
        neighbors = node_cfg.get("neighbors", [])

        neighbor_ips = {
            nid: global_cfg[nid]["ip"]
            for nid in neighbors
            if nid in global_cfg and "ip" in global_cfg[nid]
        }

        # ------------------------------------------------------------------
        # Sync-Config (global, optional)
        # ------------------------------------------------------------------
        sync_cfg = global_cfg.get("sync", {})

        # Sync module: handles mesh_time(), beacons, offsets etc.
        self.sync = SyncModule(
            node_id=node_id,
            neighbors=neighbors,
            neighbor_ips=neighbor_ips,
            sync_cfg=sync_cfg,
        )

        # ------------------------------------------------------------------
        # Hardware abstractions (currently dummy; later Grove)
        # ------------------------------------------------------------------
        self.sensor = DummySensor(sensor_type=node_cfg.get("sensor_type", "dummy"))
        self.led = DummyLED(pin=node_cfg.get("led_pin", 18))

        # Routing parent for multi-hop data (used later)
        self.parent = node_cfg.get("parent")

        # Optional disturbance button (e.g. on node D)
        self.button_pin = node_cfg.get("button_pin")

        # Stop flag if ever needed
        self._stop = asyncio.Event()

        # CoAP server + client contexts
        self._coap_ctx = None  # type: Optional[aiocoap.Context]
        self._coap_client_ctx = None  # type: Optional[aiocoap.Context]

    # ----------------------------------------------------------------------
    # Async loops
    # ----------------------------------------------------------------------

    async def sync_loop(self):
        """
        Periodically send sync beacons.
        On Windows, send_beacons() is effectively a no-op (guarded in SyncModule),
        so the loop still runs but doesn’t touch the network.
        """
        print("[{}] sync_loop started".format(self.id))
        while not self._stop.is_set():
            if self._coap_client_ctx is not None:
                await self.sync.send_beacons(self._coap_client_ctx)
            await asyncio.sleep(1.0)

    async def sensor_loop(self):
        """
        Sample the sensor periodically and (later) forward readings.
        For now, just print them with mesh timestamps.
        """
        print("[{}] sensor_loop started".format(self.id))
        while not self._stop.is_set():
            value = self.sensor.read()
            t_mesh = self.sync.mesh_time()
            print(
                "[{}] sensor reading: value={:.3f}, t_mesh={:.3f}".format(
                    self.id, value, t_mesh
                )
            )
            await asyncio.sleep(1.0)

    async def led_loop(self):
        """
        Update LED blink based on mesh time every 10 ms.
        """
        print("[{}] led_loop started".format(self.id))
        while not self._stop.is_set():
            t_mesh = self.sync.mesh_time()
            self.led.update(t_mesh)
            await asyncio.sleep(0.01)

    async def coap_loop(self):
        """
        Start the CoAP server.

        On Windows dev machines, aiocoap may not support binding to any-address;
        in that case we just skip the server so the rest of the node can run.
        """
        if IS_WINDOWS:
            print("[{}] Windows dev mode: skipping CoAP server startup".format(self.id))
            # Just sleep forever; keeps the task alive
            while True:
                await asyncio.sleep(3600)

        # Non-Windows (Pi / Linux): actually start the server
        site = build_site(self)

        try:
            # Let aiocoap choose the right bind config for the platform
            self._coap_ctx = await aiocoap.Context.create_server_context(site)
            print("[{}] CoAP server started on udp/5683".format(self.id))
            await asyncio.Future()  # run forever
        except Exception as e:
            print("[{}] Failed to start CoAP server: {}".format(self.id, e))
            # Fallback: keep going without CoAP, but don't crash the node
            while True:
                await asyncio.sleep(3600)

    async def run_async(self):
        """
        Start all node tasks.
        """
        print("[{}] MeshNode starting with cfg: {}".format(self.id, self.cfg))

        # --------------------------------------------------------------
        # 1) Create CoAP CLIENT context (for outgoing POSTs)
        # --------------------------------------------------------------
        try:
            self._coap_client_ctx = await aiocoap.Context.create_client_context()
            print("[{}] CoAP client context created".format(self.id))
        except Exception as e:
            print("[{}] Failed to create CoAP client context: {}".format(self.id, e))
            self._coap_client_ctx = None

        # --------------------------------------------------------------
        # 2) Launch all async loops
        # --------------------------------------------------------------
        await asyncio.gather(
            self.coap_loop(),    # CoAP server (no-op on Windows)
            self.sync_loop(),    # Sync beacons
            self.sensor_loop(),  # Sensor sampling
            self.led_loop(),     # LED blinking
        )

    def run(self):
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print("[{}] shutting down (KeyboardInterrupt)".format(self.id))
