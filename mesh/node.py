# -*- coding: utf-8 -*-
# mesh/node.py
"""
MeshNode — drop-in, aufgeräumt

Ziele:
- klare Verantwortlichkeiten (Sensor/LED/Storage/Sync/CoAP/Loops)
- Sync-Config sauber mergen (global -> node overrides)
- deterministisches Beacon-Slotting bleibt
- CoAP Client Context: NICHT pro Tick neu bauen (sonst jitter/overhead) → persistent
- ntp_monitor: optional lokal loggen + optional an Sink senden (persistenter client)
- Windows dev-mode: CoAP Server optional überspringen, loops laufen trotzdem

Wichtig:
- Dieses File erwartet weiterhin:
    - .sync.SyncModule (PI oder Kalman/anderer Modus via config in sync.py)
    - .coap_endpoints.build_site(node)
    - .storage.Storage.insert_ntp_reference(...)
"""

from __future__ import annotations

import asyncio
import json
import math
import platform
import time
import zlib
from pathlib import Path
from typing import Any, Dict, Optional

import aiocoap

from .coap_endpoints import build_site
from .led import DummyLED, GrovePiLED
from .sensor import DummySensor
from .storage import Storage
from .sync import SyncModule

IS_WINDOWS = platform.system() == "Windows"


class MeshNode:
    def __init__(self, node_id: str, node_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> None:
        self.id = node_id
        self.cfg = node_cfg or {}
        self.global_cfg = global_cfg or {}

        # --- Stop flag ---
        self._stop = asyncio.Event()

        # --- Sensor ---
        self.sensor = DummySensor(sensor_type=self.cfg.get("sensor_type", "dummy"))

        # --- LED ---
        self.led = self._init_led()

        # --- Optional Storage (typisch nur C) ---
        self.storage = self._init_storage()

        # --- Neighbors & IP map ---
        self.neighbors = list(self.cfg.get("neighbors", []) or [])
        self.neighbor_ips = self._build_neighbor_ip_map(self.neighbors)

        # --- Sync config merge: global defaults + node overrides ---
        sync_cfg = self._merged_sync_cfg()
        self.sync = SyncModule(
            node_id=self.id,
            neighbors=self.neighbors,
            neighbor_ips=self.neighbor_ips,
            sync_cfg=sync_cfg,
            storage=self.storage,
        )

        # --- Routing / telemetry ---
        self.parent = self.cfg.get("parent")
        self.telemetry_sink_id = self.cfg.get("telemetry_sink") or self.parent or "C"
        self.telemetry_sink_ip = self._resolve_ip(self.telemetry_sink_id)

        # --- Optional button (not used here, but kept) ---
        self.button_pin = self.cfg.get("button_pin")

        # --- CoAP contexts ---
        self._coap_server_ctx: Optional[aiocoap.Context] = None
        self._coap_client_ctx: Optional[aiocoap.Context] = None

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_led(self):
        led_pin = self.cfg.get("led_pin", None)
        if led_pin is None:
            print(f"[{self.id}] no led_pin configured → LED disabled")
            return None

        if IS_WINDOWS:
            return DummyLED(pin=led_pin)

        try:
            return GrovePiLED(pin=led_pin)
        except Exception as e:
            print(f"[{self.id}] GrovePiLED failed ({e}) → falling back to DummyLED")
            return DummyLED(pin=led_pin)

    def _init_storage(self) -> Optional[Storage]:
        db_path = self.cfg.get("db_path")
        if not db_path:
            return None
        try:
            print(f"[{self.id}] Initializing local DB at {db_path}")
            return Storage(db_path)
        except Exception as e:
            print(f"[{self.id}] Storage init failed ({e}) → DB disabled")
            return None

    def _resolve_ip(self, node_id: Optional[str]) -> Optional[str]:
        if not node_id or node_id == self.id:
            return None
        entry = self.global_cfg.get(node_id) or {}
        return entry.get("ip")

    def _build_neighbor_ip_map(self, neighbors):
        ips: Dict[str, str] = {}
        for nid in neighbors:
            ip = self._resolve_ip(nid)
            if ip:
                ips[nid] = ip
        return ips

    def _merged_sync_cfg(self) -> Dict[str, Any]:
        base = (self.global_cfg.get("sync", {}) or {})
        local = (self.cfg.get("sync", {}) or {})
        merged = dict(base)
        merged.update(local)
        return merged

    # ------------------------------------------------------------------
    # CoAP contexts
    # ------------------------------------------------------------------

    async def _ensure_client_ctx(self) -> Optional[aiocoap.Context]:
        if IS_WINDOWS:
            return None
        if self._coap_client_ctx is None:
            self._coap_client_ctx = await aiocoap.Context.create_client_context()
        return self._coap_client_ctx

    async def _shutdown_client_ctx(self) -> None:
        if self._coap_client_ctx is not None:
            try:
                await self._coap_client_ctx.shutdown()
            except Exception:
                pass
            self._coap_client_ctx = None

    # ------------------------------------------------------------------
    # Loops
    # ------------------------------------------------------------------

    async def sync_loop(self) -> None:
        """
        Periodically send sync beacons (deterministic slotting).
        Uses a persistent CoAP client context to reduce jitter/overhead.
        """
        print(f"[{self.id}] sync_loop started")

        sync_global = (self.global_cfg.get("sync", {}) or {})
        slot_ms = float(sync_global.get("beacon_slot_ms", 50.0))
        n_slots = int(sync_global.get("beacon_n_slots", 8))
        period_s = float(sync_global.get("beacon_period_s", 1.0))

        h = zlib.crc32(self.id.encode("utf-8")) & 0xFFFFFFFF
        slot_idx = h % max(1, n_slots)
        phase_s = (slot_idx * slot_ms) / 1000.0

        print(
            f"[{self.id}] beacon slotting: slot_idx={slot_idx}/{n_slots}, "
            f"phase={phase_s*1000:.1f}ms, period={period_s:.2f}s"
        )

        while not self._stop.is_set():
            if IS_WINDOWS:
                await asyncio.sleep(1.0)
                continue

            now = time.monotonic()
            t0 = math.floor(now / period_s) * period_s
            target = t0 + phase_s
            if target <= now:
                target += period_s

            await asyncio.sleep(max(0.0, target - now))

            try:
                ctx = await self._ensure_client_ctx()
                if ctx is None:
                    continue
                await self.sync.send_beacons(ctx)
            except Exception as e:
                print(f"[{self.id}] sync_loop: send_beacons failed: {e}")

        # cleanup
        await self._shutdown_client_ctx()

    async def sensor_loop(self) -> None:
        print(f"[{self.id}] sensor_loop started")
        while not self._stop.is_set():
            value = self.sensor.read()
            t_mesh = self.sync.mesh_time()
            print(f"[{self.id}] sensor reading: value={value:.3f}, t_mesh={t_mesh:.3f}")
            await asyncio.sleep(1.0)

    async def led_loop(self) -> None:
        print(f"[{self.id}] led_loop started")
        while not self._stop.is_set():
            if self.led is not None:
                self.led.update(self.sync.mesh_time())
            await asyncio.sleep(0.01)

    async def coap_loop(self) -> None:
        """
        Start the CoAP server. On Windows dev mode: skip server.
        """
        if IS_WINDOWS:
            print(f"[{self.id}] Windows dev mode: skipping CoAP server startup")
            while not self._stop.is_set():
                await asyncio.sleep(3600)
            return

        site = build_site(self)
        try:
            self._coap_server_ctx = await aiocoap.Context.create_server_context(site)
            print(f"[{self.id}] CoAP server started (aiocoap default bind)")
        except Exception as e:
            print(f"[{self.id}] CoAP server failed to start: {e}")
            raise

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass

    async def ntp_monitor_loop(self, interval: float = 5.0) -> None:
        print(f"[{self.id}] ntp_monitor_loop started (interval={interval}s)")

        while not self._stop.is_set():
            t_wall = time.time()
            t_mono = time.monotonic()
            t_mesh = self.sync.mesh_time()
            offset = self.sync.get_offset()
            err = t_mesh - t_wall

            # 1) Local DB: IMMER loggen
            if self.storage is not None:
                try:
                    self.storage.insert_ntp_reference(
                        node_id=self.id,
                        t_wall=t_wall,
                        t_mono=t_mono,
                        t_mesh=t_mesh,
                        offset=offset,
                        err_mesh_vs_wall=err,
                        peer_id=None,
                        theta_ms=None,
                        rtt_ms=None,
                        sigma_ms=None,
                    )
                except Exception as e:
                    print(f"[{self.id}] ntp_monitor_loop: local DB insert failed: {e}")

            # 2) Sink: optional erst nach Warmup
            if self.sync.is_warmed_up() and self.telemetry_sink_ip is not None:
                try:
                    ctx = await self._ensure_client_ctx()
                    if ctx is not None:
                        uri = f"coap://{self.telemetry_sink_ip}/relay/ingest/ntp"
                        payload = {
                            "node_id": self.id,
                            "t_wall": t_wall,
                            "t_mono": t_mono,
                            "t_mesh": t_mesh,
                            "offset": offset,
                            "err_mesh_vs_wall": err,
                        }
                        req = aiocoap.Message(
                            code=aiocoap.POST,
                            uri=uri,
                            payload=json.dumps(payload).encode("utf-8"),
                        )
                        await asyncio.wait_for(ctx.request(req).response, timeout=0.5)
                except Exception as e:
                    print(f"[{self.id}] ntp_monitor_loop: CoAP failed: {e}")

            await asyncio.sleep(interval)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run_async(self) -> None:
        print(f"[{self.id}] MeshNode starting with cfg: {self.cfg}")

        tasks = []
        for coro, name in [
            (self.coap_loop(), f"{self.id}:coap"),
            (self.sync_loop(), f"{self.id}:sync"),
            (self.sensor_loop(), f"{self.id}:sensor"),
            (self.led_loop(), f"{self.id}:led"),
            (self.ntp_monitor_loop(), f"{self.id}:ntp"),
        ]:
            t = asyncio.create_task(coro)
            # Python >=3.8 only — safe guard for 3.7
            if hasattr(t, "set_name"):
                t.set_name(name)
            tasks.append(t)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self._stop.set()

            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            await self._shutdown_client_ctx()
            if self._coap_server_ctx is not None:
                try:
                    await self._coap_server_ctx.shutdown()
                except Exception:
                    pass
                self._coap_server_ctx = None

    def run(self) -> None:
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            print(f"[{self.id}] shutting down (KeyboardInterrupt)")
