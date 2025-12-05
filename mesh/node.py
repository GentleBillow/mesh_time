# -*- coding: utf-8 -*-
# mesh/node.py

import asyncio
import platform
from typing import Dict, Any

import aiocoap

from .sync import SyncModule
from .sensor import DummySensor
from .led import DummyLED
from .coap_endpoints import build_site

IS_WINDOWS = platform.system() == "Windows"


class MeshNode:
    def __init__(self, node_id, node_cfg, global_cfg):
        self.id = node_id
        self.cfg = node_cfg
        self.global_cfg = global_cfg

        # ------------------------------------------------------------------
        # Neighbors & their IPs (from global config)
        # ------------------------------------------------------------------
        neighbors = node_cfg.get("neighbors", [])

        neighbor_ips = {}
        for nid in neighbors:
            if nid in global_cfg and "ip" in global_cfg[nid]:
                neighbor_ips[nid] = global_cfg[nid]["ip"]

        # Sync module: handles mesh_time(), beacons, offsets etc.
        self.sync = SyncModule(
            node_id=node_id,
            neighbors=neighbors,
            neighbor_ips=neighbor_ips,
            sync_cfg=global_cfg.get("sync", {}),
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

        # CoAP server context (for incoming requests)
        self._coap_ctx = None

    # ----------------------------------------------------------------------
    # Async loops
    # ----------------------------------------------------------------------

    async def sync_loop(self):
        """
        Periodically send sync beacons.

        Design choice:
          - We create a fresh aiocoap client context on each iteration.
          - This avoids issues where a single ICMP "port unreachable" or
            other socket-level error poisons a long-lived context.
        """
        print("[{}] sync_loop started".format(self.id))
        while not self._stop.is_set():
            if IS_WINDOWS:
                # Dev mode: no network sync on Windows
                await asyncio.sleep(1.0)
                continue

            try:
                client_ctx = await aiocoap.Context.create_client_context()
            except Exception as e:
                print("[{}] sync_loop: failed to create client context: {}".format(self.id, e))
                await asyncio.sleep(1.0)
                continue

            try:
                await self.sync.send_beacons(client_ctx)
            except Exception as e:
                print("[{}] sync_loop: error in send_beacons: {}".format(self.id, e))

            # Cleanly shut down the client context so sockets are released
            try:
                await client_ctx.shutdown()
            except Exception:
                pass

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
            print("[{}] sensor reading: value={:.3f}, t_mesh={:.3f}".format(
                self.id, value, t_mesh
            ))
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
            while True:
                await asyncio.sleep(3600)

        site = build_site(self)

        try:
            # Explicitly bind to IPv4 on all interfaces
            self._coap_ctx = await aiocoap.Context.create_server_context(
                site,
                bind=("0.0.0.0", 5683),
            )
            print("[{}] CoAP server started on udp/5683".format(self.id))
            await asyncio.Future()  # run forever
        except Exception as e:
            print("[{}] Failed to start CoAP server: {}".format(self.id, e))
            while True:
                await asyncio.sleep(3600)

    async def run_async(self):
        """
        Start all node tasks.
        """
        print("[{}] MeshNode starting with cfg: {}".format(self.id, self.cfg))

        await asyncio.gather(
            self.coap_loop(),    # CoAP server
            self.sync_loop(),    # Sync beacons
            self.sensor_loop(),  # Sensor sampling
            self.led_loop(),     # LED blinking
        )

    def run(self):
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print("[{}] shutting down (KeyboardInterrupt)".format(self.id))
