# -*- coding: utf-8 -*-
# mesh/node.py

import asyncio
import time
import platform
from typing import Dict, Any

import aiocoap

from .sync import SyncModule
from .sensor import DummySensor
from .led import DummyLED, GrovePiLED
from .coap_endpoints import build_site
from .storage import Storage

IS_WINDOWS = platform.system() == "Windows"


class MeshNode:
    def __init__(self, node_id, node_cfg, global_cfg):
        self.id = node_id
        self.cfg = node_cfg
        self.global_cfg = global_cfg

        # ------------------------------------------------------------------
        # Sensor
        # ------------------------------------------------------------------
        self.sensor = DummySensor(sensor_type=node_cfg.get("sensor_type", "dummy"))

        # ------------------------------------------------------------------
        # LED
        # ------------------------------------------------------------------
        led_pin = node_cfg.get("led_pin", None)
        if led_pin is None:
            print(f"[{node_id}] no led_pin configured → LED disabled")
            self.led = None
        else:
            if IS_WINDOWS:
                # Dev-Mode
                self.led = DummyLED(pin=led_pin)
            else:
                try:
                    self.led = GrovePiLED(pin=led_pin)
                except Exception as e:
                    print(f"[{node_id}] GrovePiLED failed ({e}) → falling back to DummyLED")
                    self.led = DummyLED(pin=led_pin)

        # ------------------------------------------------------------------
        # Optional lokale DB (z.B. nur auf Node C)
        # ------------------------------------------------------------------
        self.storage = None
        db_path = node_cfg.get("db_path")
        if db_path:
            print(f"[{self.id}] Initializing local DB at {db_path}")
            self.storage = Storage(db_path)

        # ------------------------------------------------------------------
        # Neighbors & their IPs (from global config)
        # ------------------------------------------------------------------
        neighbors = node_cfg.get("neighbors", [])

        neighbor_ips: Dict[str, str] = {}
        for nid in neighbors:
            if nid in global_cfg and "ip" in global_cfg[nid]:
                neighbor_ips[nid] = global_cfg[nid]["ip"]

        # ------------------------------------------------------------------
        # Sync-Konfiguration: global + node-spezifisch mergen
        # ------------------------------------------------------------------
        base_sync_cfg = global_cfg.get("sync", {}) or {}
        node_sync_cfg = node_cfg.get("sync", {}) or {}

        sync_cfg = dict(base_sync_cfg)   # Kopie der globalen Defaults
        sync_cfg.update(node_sync_cfg)   # node-spezifische Overrides (z.B. is_root)

        # Sync module: handles mesh_time(), beacons, offsets etc.
        self.sync = SyncModule(
            node_id=node_id,
            neighbors=neighbors,
            neighbor_ips=neighbor_ips,
            sync_cfg=sync_cfg,
        )

        # Routing parent für spätere Multi-Hop-Daten
        self.parent = node_cfg.get("parent")

        # Optional disturbance button (z.B. auf Node D)
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
            if self.led is not None:
                self.led.update(t_mesh)
            await asyncio.sleep(0.01)

    async def coap_loop(self):
        """
        Start the CoAP server.
        """
        if IS_WINDOWS:
            print("[{}] Windows dev mode: skipping CoAP server startup".format(self.id))
            while True:
                await asyncio.sleep(3600)

        site = build_site(self)

        try:
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

    async def ntp_monitor_loop(self, interval: float = 5.0):
        """
        Periodisch NTP-Referenzwerte loggen.
        """
        if self.storage is None:
            while True:
                await asyncio.sleep(3600.0)

        print(f"[{self.id}] ntp_monitor_loop started (interval={interval}s)")
        while True:
            t_wall = time.time()
            t_mono = time.monotonic()
            t_mesh = self.sync.mesh_time()
            offset = self.sync.get_offset()
            err = t_mesh - t_wall

            try:
                self.storage.insert_ntp_reference(
                    node_id=self.id,
                    t_wall=t_wall,
                    t_mono=t_mono,
                    t_mesh=t_mesh,
                    offset=offset,
                    err_mesh_vs_wall=err,
                )
            except Exception as e:
                print(f"[{self.id}] ntp_monitor_loop: DB insert failed: {e}")

            await asyncio.sleep(interval)

    async def run_async(self):
        """
        Start all node tasks.
        """
        print("[{}] MeshNode starting with cfg: {}".format(self.id, self.cfg))

        tasks = [
            self.coap_loop(),    # CoAP server
            self.sync_loop(),    # Sync beacons
            self.sensor_loop(),  # Sensor sampling
            self.led_loop(),     # LED blinking
        ]

        # Nur Knoten mit DB (z.B. C) loggen NTP-Referenz
        if self.storage is not None:
            tasks.append(self.ntp_monitor_loop())

        await asyncio.gather(*tasks)

    def run(self):
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print("[{}] shutting down (KeyboardInterrupt)".format(self.id))
