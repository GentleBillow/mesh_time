# -*- coding: utf-8 -*-
# mesh/node.py

import asyncio
import time
import platform
import json
from typing import Dict, Any
import zlib
import math


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
        # Optional lokale DB (typisch nur auf Node C)
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
            storage=self.storage,  # <--- wichtig: Storage an SyncModule durchreichen
        )

        # Routing parent für spätere Multi-Hop-Daten / Telemetrie
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

        # --- Beacon Slotting (deterministisch) ---
        slot_ms = float(self.global_cfg.get("sync", {}).get("beacon_slot_ms", 50.0))  # z.B. 50ms
        n_slots = int(self.global_cfg.get("sync", {}).get("beacon_n_slots", 8))  # z.B. 8 Slots
        period_s = float(self.global_cfg.get("sync", {}).get("beacon_period_s", 1.0))  # z.B. 1s

        # stabile "Hash"-Funktion (crc32), damit über Reboots konstant
        h = zlib.crc32(self.id.encode("utf-8")) & 0xffffffff
        slot_idx = h % n_slots
        phase_s = (slot_idx * slot_ms) / 1000.0

        print(f"[{self.id}] beacon slotting: slot_idx={slot_idx}/{n_slots}, phase={phase_s * 1000:.1f}ms")

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

            # Slot-Wait: sende in deinem festen Zeitfenster innerhalb jeder Periode
            now = time.monotonic()
            t0 = int(now / period_s) * period_s
            target = t0 + phase_s
            if target <= now:
                target += period_s

            await asyncio.sleep(max(0.0, target - now))

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
        Periodisch Mesh-Zeit-Telemetrie loggen.

        Wir benutzen time.time() hier nur als gemeinsame X-Achse
        (kein echter NTP-Server nötig):

            t_wall  = time.time()
            t_mono  = time.monotonic()
            t_mesh  = sync.mesh_time()
            offset  = sync.get_offset()
            err     = t_mesh - t_wall

        - Wenn self.storage vorhanden (typisch Node C):
              -> direkt in lokale DB schreiben.
        - Zusätzlich: per CoAP an Sammelknoten (parent oder explizit 'C')
              POST /relay/ingest/ntp
        """
        print(f"[{self.id}] ntp_monitor_loop started (interval={interval}s)")

        # Ziel-Knoten für Telemetrie bestimmen: parent oder explizit 'C'
        sink_id = self.cfg.get("telemetry_sink") or self.parent or "C"
        sink_ip = None
        if sink_id != self.id and sink_id in self.global_cfg and "ip" in self.global_cfg[sink_id]:
            sink_ip = self.global_cfg[sink_id]["ip"]

        while not self._stop.is_set():
            t_wall = time.time()
            t_mono = time.monotonic()
            t_mesh = self.sync.mesh_time()
            offset = self.sync.get_offset()
            err = t_mesh - t_wall

            # 1) Lokal in DB loggen (nur auf Node mit Storage, z.B. C)
            if self.storage is not None:
                try:
                    self.storage.insert_ntp_reference(
                        node_id=self.id,
                        t_wall=t_wall,
                        t_mono=t_mono,
                        t_mesh=t_mesh,
                        offset=offset,
                        err_mesh_vs_wall=err,
                        # alle Link-spezifischen Felder lässt Storage
                        # einfach default-mäßig auf None
                    )
                except Exception as e:
                    print(f"[{self.id}] ntp_monitor_loop: local DB insert failed: {e}")

            # 2) An Sammelknoten schicken (typisch C)
            if sink_ip is not None:
                uri = f"coap://{sink_ip}/relay/ingest/ntp"
                payload = {
                    "node_id": self.id,
                    "t_wall": t_wall,
                    "t_mono": t_mono,
                    "t_mesh": t_mesh,
                    "offset": offset,
                    "err_mesh_vs_wall": err,
                }
                try:
                    req = aiocoap.Message(
                        code=aiocoap.POST,
                        uri=uri,
                        payload=json.dumps(payload).encode("utf-8"),
                    )
                    ctx = await aiocoap.Context.create_client_context()
                    # Fire-and-forget; Timeout klein halten
                    await asyncio.wait_for(ctx.request(req).response, timeout=0.5)
                    await ctx.shutdown()
                except asyncio.TimeoutError:
                    print(f"[{self.id}] ntp_monitor_loop: CoAP to {sink_id} timed out")
                except Exception as e:
                    print(f"[{self.id}] ntp_monitor_loop: CoAP to {sink_id} failed: {e}")

            await asyncio.sleep(interval)

    async def run_async(self):
        """
        Start all node tasks.
        """
        print("[{}] MeshNode starting with cfg: {}".format(self.id, self.cfg))

        tasks = [
            self.coap_loop(),        # CoAP server
            self.sync_loop(),        # Sync beacons
            self.sensor_loop(),      # Sensor sampling
            self.led_loop(),         # LED blinking
            self.ntp_monitor_loop(), # Telemetrie für Web-UI
        ]

        await asyncio.gather(*tasks)

    def run(self):
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print("[{}] shutting down (KeyboardInterrupt)".format(self.id))
