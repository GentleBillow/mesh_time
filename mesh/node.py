# -*- coding: utf-8 -*-
# mesh/node.py - ROBUST VERSION (Py3.7 safe, no cross-loop primitives)

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

        # Logging config (idempotent-ish, OK for your project usage)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # NOTE: NEVER create asyncio primitives (Event/Queue/Lock) here.
        # They would bind to whichever loop happens to be current, and Py3.7 will bite you.
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Event] = None
        self._coap_ready: Optional[asyncio.Event] = None

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
    # CoAP client context
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
        """Start sync workers and wait for stop."""
        assert self._stop is not None

        log.info("[%s] sync_loop started", self.id)

        if IS_WINDOWS:
            log.info("[%s] Windows dev mode: skipping sync workers", self.id)
            await self._stop.wait()
            return

        ctx = await self._ensure_client_ctx()
        if ctx is None:
            return

        try:
            await self.sync.start(ctx)
            await self._stop.wait()
        finally:
            await self.sync.stop()
            await self._shutdown_client_ctx()

    async def sensor_loop(self) -> None:
        assert self._stop is not None
        log.info("[%s] sensor_loop started", self.id)

        while not self._stop.is_set():
            _ = self.sensor.read()
            _ = self.sync.mesh_time()
            await asyncio.sleep(1.0)

    async def led_loop(self) -> None:
        assert self._stop is not None
        log.info("[%s] led_loop started", self.id)

        while not self._stop.is_set():
            if self.led is not None:
                self.led.update(self.sync.mesh_time())
            await asyncio.sleep(0.01)

    async def ntp_monitor_loop(self, interval: float = 5.0) -> None:
        """Periodic telemetry + optional local DB logging."""
        assert self._stop is not None

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

            # 2) Telemetry (best effort)
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
                            "warmed_up": bool(self.sync.is_warmed_up()),
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

            await asyncio.sleep(interval)

    async def coap_loop(self) -> None:
        """Start CoAP server with hard timeout; always signals _coap_ready."""
        assert self._stop is not None
        assert self._coap_ready is not None

        if IS_WINDOWS:
            log.info("[%s] Windows dev mode: skipping CoAP server", self.id)
            self._coap_ready.set()
            await self._stop.wait()
            return

        site = build_site(self)

        bind_ip = "0.0.0.0"
        bind_port = 5683

        task: Optional[asyncio.Task] = None
        try:
            log.info("[%s] CoAP server: creating server context… bind=%s:%d", self.id, bind_ip, bind_port)

            # explicit task so we can cancel it on timeout safely
            task = asyncio.create_task(
                aiocoap.Context.create_server_context(site, bind=(bind_ip, bind_port))
            )

            self._coap_server_ctx = await asyncio.wait_for(task, timeout=3.0)
            log.info("[%s] CoAP server started (bind=%s:%d)", self.id, bind_ip, bind_port)

        except asyncio.TimeoutError:
            log.error("[%s] CoAP server start TIMEOUT (bind=%s:%d)", self.id, bind_ip, bind_port)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise

        except asyncio.CancelledError:
            # normal during shutdown if run_async cancels us
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise

        except Exception as e:
            log.exception("[%s] CoAP server failed to start: %s", self.id, e)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            raise

        finally:
            # ALWAYS unblock run_async (success or failure)
            self._coap_ready.set()

        # Keep alive until stop; shutdown is handled in run_async finally
        await self._stop.wait()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    async def run_async(self) -> None:
        self._loop = asyncio.get_running_loop()

        # Create asyncio primitives ONLY here (bound to the correct running loop)
        self._stop = asyncio.Event()
        self._coap_ready = asyncio.Event()

        log.info("[%s] MeshNode starting with cfg: %s", self.id, self.cfg)

        tasks = []

        # 1) Start CoAP first
        coap_task = asyncio.create_task(self.coap_loop())
        if hasattr(coap_task, "set_name"):
            coap_task.set_name(f"{self.id}:coap")
        tasks.append(coap_task)

        # 2) Wait until CoAP signals ready (or timeout); proceed either way
        try:
            await asyncio.wait_for(self._coap_ready.wait(), timeout=4.0)
        except asyncio.TimeoutError:
            log.warning("[%s] CoAP ready wait TIMEOUT – continuing anyway", self.id)

        # 3) Start remaining loops
        for coro, name in [
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
            # Signal stop first
            if self._stop is not None:
                self._stop.set()

            # Cancel all tasks
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

            # Shutdown client ctx
            await self._shutdown_client_ctx()

            # Shutdown server ctx
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
