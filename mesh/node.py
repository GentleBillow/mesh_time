# -*- coding: utf-8 -*-
# mesh/node.py - ROBUST VERSION für sync_robust.py

from __future__ import annotations

import asyncio
import json
import logging
import platform
import time
from typing import Any, Dict, Optional

import aiocoap

from .coap_endpoints import build_site
from .led import DummyLED, GrovePiLED
from .sensor import DummySensor
from .storage import Storage
from .sync import SyncModule

IS_WINDOWS = platform.system() == "Windows"

log = logging.getLogger("meshtime.node")


class MeshNode:
    def __init__(self, node_id: str, node_cfg: Dict[str, Any], global_cfg: Dict[str, Any]) -> None:
        self.id = node_id
        self.cfg = node_cfg or {}
        self.global_cfg = global_cfg or {}

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # --- Stop flag ---
        self._stop = asyncio.Event()

        # --- Sensor ---
        self.sensor = DummySensor(sensor_type=self.cfg.get("sensor_type", "dummy"))

        # --- LED ---
        self.led = self._init_led()

        # --- Optional Storage ---
        self.storage = self._init_storage()

        # --- Neighbors & IP map ---
        self.neighbors = list(self.cfg.get("neighbors", []) or [])
        self.neighbor_ips = self._build_neighbor_ip_map(self.neighbors)

        # --- Routing / telemetry ---
        self.parent = self.cfg.get("parent")
        self.telemetry_sink_id = self.cfg.get("telemetry_sink") or self.parent or "C"
        self.telemetry_sink_ip = self._resolve_ip(self.telemetry_sink_id)

        # --- Sync config merge ---
        sync_cfg = self._merged_sync_cfg()

        # ROBUST: Direkte IP-Übergabe
        telemetry_ip = None
        if self.storage is None:
            telemetry_ip = self.telemetry_sink_ip

        self.sync = SyncModule(
            node_id=self.id,
            neighbors=self.neighbors,
            neighbor_ips=self.neighbor_ips,
            sync_cfg=sync_cfg,
            storage=self.storage,
            telemetry_sink_ip=telemetry_ip,
        )

        # --- CoAP contexts ---
        self._coap_server_ctx: Optional[aiocoap.Context] = None
        self._coap_client_ctx: Optional[aiocoap.Context] = None

    # ------------------------------------------------------------------
    # Init helpers
    # ------------------------------------------------------------------

    def _init_led(self):
        led_pin = self.cfg.get("led_pin", None)
        if led_pin is None:
            log.info("[%s] No led_pin configured → LED disabled", self.id)
            return None

        if IS_WINDOWS:
            return DummyLED(pin=led_pin)

        try:
            return GrovePiLED(pin=led_pin)
        except Exception as e:
            log.warning("[%s] GrovePiLED failed (%s) → falling back to DummyLED", self.id, e)
            return DummyLED(pin=led_pin)

    def _init_storage(self) -> Optional[Storage]:
        db_path = self.cfg.get("db_path")
        if not db_path:
            return None
        try:
            log.info("[%s] Initializing local DB at %s", self.id, db_path)
            return Storage(db_path)
        except Exception as e:
            log.error("[%s] Storage init failed (%s) → DB disabled", self.id, e)
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
    # ROBUST: Sync Loop (simplified)
    # ------------------------------------------------------------------

    async def sync_loop(self) -> None:
        """
        ROBUST: Startet einfach nur die Sync-Worker.
        Kein slotting mehr nötig - jeder Peer-Worker managed sein Timing.
        """
        log.info("[%s] sync_loop started", self.id)

        if IS_WINDOWS:
            log.info("[%s] Windows dev mode: skipping sync workers", self.id)
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
            return

        # Get client context
        ctx = await self._ensure_client_ctx()
        if ctx is None:
            return

        try:
            # ROBUST: Start all peer workers
            await self.sync.start(ctx)

            # Wait until stop
            await self._stop.wait()

        finally:
            # ROBUST: Stop all workers
            await self.sync.stop()
            await self._shutdown_client_ctx()

    # ------------------------------------------------------------------
    # Other loops (unchanged)
    # ------------------------------------------------------------------

    async def sensor_loop(self) -> None:
        log.info("[%s] sensor_loop started", self.id)
        while not self._stop.is_set():
            value = self.sensor.read()
            t_mesh = self.sync.mesh_time()
            log.debug("[%s] sensor reading: value=%.3f, t_mesh=%.3f", self.id, value, t_mesh)
            await asyncio.sleep(1.0)

    async def led_loop(self) -> None:
        log.info("[%s] led_loop started", self.id)
        while not self._stop.is_set():
            if self.led is not None:
                self.led.update(self.sync.mesh_time())
            await asyncio.sleep(0.01)

    async def coap_loop(self) -> None:
        """Start the CoAP server."""
        if IS_WINDOWS:
            log.info("[%s] Windows dev mode: skipping CoAP server", self.id)
            while not self._stop.is_set():
                await asyncio.sleep(3600)
            return

        site = build_site(self)

        # FIX: bind to the node's configured IPv4 (avoids aiocoap hanging on 0.0.0.0)
        bind_ip = (self.cfg.get("ip") or "0.0.0.0")
        bind_port = 5683

        log.info("[%s] CoAP server: creating server context… bind=%s:%d", self.id, bind_ip, bind_port)

        # Make it an explicit task so we can cancel without deadlocking on cancellation
        task = asyncio.create_task(
            aiocoap.Context.create_server_context(site, bind=(bind_ip, bind_port))
        )

        try:
            self._coap_server_ctx = await asyncio.wait_for(task, timeout=3.0)
            log.info("[%s] CoAP server started (bind=%s:%d)", self.id, bind_ip, bind_port)

        except asyncio.TimeoutError:
            log.error("[%s] CoAP server start TIMEOUT (bind=%s:%d)", self.id, bind_ip, bind_port)
            log.error("[%s] coap task done=%s cancelled=%s", self.id, task.done(), task.cancelled())
            task.cancel()
            raise

        except asyncio.CancelledError:
            log.exception("[%s] CoAP server cancelled!", self.id)
            task.cancel()
            raise

        except Exception:
            task.cancel()
            log.exception("[%s] CoAP server failed to start", self.id)
            raise

        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass


    async def ntp_monitor_loop(self, interval: float = 5.0) -> None:
        """Loggt globalen Node-Status."""
        log.info("[%s] ntp_monitor_loop started (interval=%.1fs)", self.id, interval)

        while not self._stop.is_set():
            t_wall = time.time()
            t_mono = time.monotonic()
            t_mesh = self.sync.mesh_time()
            offset = self.sync.get_offset()
            err = t_mesh - t_wall

            # 1) Local DB
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
                    log.error("[%s] ntp_monitor_loop: DB insert failed: %s", self.id, e)


            # 2) Telemetry: IMMER senden (auch vor warmup), damit UI alle Nodes sieht.
            # Warmup ist ein Regler/Sigma-Thema, aber fürs "Node lebt" + offset Plot wollen wir Daten ab Start.
            if self.telemetry_sink_ip is not None:
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
                            "warmed_up": bool(self.sync.is_warmed_up()),  # optional, nice for debug
                        }
                        req = aiocoap.Message(
                            code=aiocoap.POST,
                            uri=uri,
                            payload=json.dumps(payload).encode("utf-8"),
                        )
                        await asyncio.wait_for(ctx.request(req).response, timeout=0.5)
                except asyncio.TimeoutError:
                    log.warning("[%s] ntp_monitor_loop: telemetry timeout", self.id)
                except Exception as e:
                    log.warning("[%s] ntp_monitor_loop: telemetry failed: %s", self.id, type(e).__name__)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run_async(self) -> None:
        log.info("[%s] MeshNode starting with cfg: %s", self.id, self.cfg)

        tasks = []
        for coro, name in [
            (self.coap_loop(), f"{self.id}:coap"),
            (self.sync_loop(), f"{self.id}:sync"),
            (self.sensor_loop(), f"{self.id}:sensor"),
            (self.led_loop(), f"{self.id}:led"),
            (self.ntp_monitor_loop(), f"{self.id}:ntp"),
        ]:
            t = asyncio.create_task(coro)
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
            log.info("[%s] shutting down (KeyboardInterrupt)", self.id)