# -*- coding: utf-8 -*-
# mesh/sync.py - ROBUST VERSION nach Forum-Best-Practices

from __future__ import annotations

import asyncio
import json
import logging
import math
import platform
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import concurrent.futures

import aiocoap

IS_WINDOWS = platform.system() == "Windows"

# Setup logging
log = logging.getLogger("meshtime.sync")


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class PeerStats:
    """Per-peer rolling link statistics (seconds)."""
    rtt_samples: List[float] = field(default_factory=list)
    sigma: Optional[float] = None
    theta_last: Optional[float] = None
    good_samples: int = 0
    last_theta_epoch: Optional[float] = None
    rtt_last: Optional[float] = None
    rtt_baseline: Optional[float] = None
    state: str = "DOWN"  # UP or DOWN


# ---------------------------------------------------------------------
# Controller interface + implementations
# ---------------------------------------------------------------------

Measurement = Tuple[str, float, float, float]
WeightFn = Callable[[str], float]


class BaseController:
    """Controller interface."""

    def on_bootstrap(self) -> None:
        pass

    def compute_delta(
            self,
            offset_s: float,
            measurements: List[Measurement],
            weight_fn: WeightFn,
            dt_s: float,
    ) -> float:
        raise NotImplementedError


class PIController(BaseController):
    """Weighted global PI controller."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.kp = float(cfg.get("kp", 0.02))
        self.ki = float(cfg.get("ki", 0.001))
        self.i_leak = float(cfg.get("i_leak", 0.0))
        self.i_max = float(cfg.get("i_max_ms", 200.0)) / 1000.0
        self.i_state = 0.0

    def on_bootstrap(self) -> None:
        self.i_state = 0.0

    def compute_delta(
            self,
            offset_s: float,
            measurements: List[Measurement],
            weight_fn: WeightFn,
            dt_s: float,
    ) -> float:
        num = 0.0
        den = 0.0

        for peer_id, _rtt, theta_s, peer_offset_s in measurements:
            w = float(weight_fn(peer_id))
            target = peer_offset_s - theta_s
            e = offset_s - target
            num += w * e
            den += w

        if den <= 0.0:
            return 0.0

        e = num / den

        if self.i_leak > 0.0:
            self.i_state *= max(0.0, 1.0 - self.i_leak * dt_s)

        self.i_state += e * dt_s
        if self.i_max > 0.0:
            self.i_state = max(-self.i_max, min(self.i_max, self.i_state))

        return -self.kp * e - self.ki * self.i_state


# ---------------------------------------------------------------------
# ROBUST: Task spawn helper (Python 3.7 compatible)
# ---------------------------------------------------------------------

def spawn_task(coro, name: str) -> asyncio.Task:
    """
    Spawn an asyncio task with:
      - Python 3.7 compatibility (no name= kwarg)
      - explicit exception logging (no silent failures)
      - graceful cancellation handling

    Behaviour:
      - CancelledError → INFO
      - Any other exception → ERROR + traceback
    """

    # Python 3.7: no `name=` argument
    task = asyncio.create_task(coro)

    # Set task name if supported (Py ≥ 3.8)
    if hasattr(task, "set_name"):
        try:
            task.set_name(name)
        except Exception:
            pass

    def _done_callback(t: asyncio.Task):
        try:
            t.result()
        except asyncio.CancelledError:
            log.info("Task cancelled: %s", name)
        except Exception:
            # This is CRITICAL: never swallow task crashes
            log.exception("Task CRASHED: %s", name)

    task.add_done_callback(_done_callback)
    return task


# ---------------------------------------------------------------------
# Sync module
# ---------------------------------------------------------------------

class SyncModule:
    """
    ROBUST mesh time synchronization.

    Key improvements:
    - Separate worker per peer (non-blocking)
    - Hard timeouts on all requests
    - Exponential backoff for down peers
    - Proper exception logging
    - No silent failures
    """

    def __init__(
            self,
            node_id: str,
            neighbors: List[str],
            neighbor_ips: Dict[str, str],
            sync_cfg: Optional[Dict[str, Any]] = None,
            storage=None,
            telemetry_sink_ip: Optional[str] = None,
    ) -> None:
        sync_cfg = sync_cfg or {}

        self.node_id = node_id
        self.neighbors = list(neighbors)
        self.neighbor_ips = dict(neighbor_ips)
        self._storage = storage
        self._telemetry_sink_ip = telemetry_sink_ip

        # role
        self._is_root = bool(sync_cfg.get("is_root", False))

        # warmup / bootstrap
        self._bootstrapped = bool(self._is_root)
        self._beacon_count = 0
        self._min_beacons_for_warmup = int(sync_cfg.get("min_beacons_for_warmup", 15))
        self._bootstrap_theta_max_s = float(sync_cfg.get("bootstrap_theta_max_s", 0.0))
        self._bootstrap_peers = set(sync_cfg.get("bootstrap_peers", []) or [])

        # time lifting
        self._boot_epoch = time.time() - time.monotonic()

        # offset init
        init_ms = float(sync_cfg.get("initial_offset_ms", 200.0))
        self._offset = 0.0 if self._is_root else random.uniform(-init_ms / 1000, init_ms / 1000)

        # controller selection
        ctrl_name = (sync_cfg.get("controller", "pi") or "pi").lower()
        if ctrl_name == "pi":
            self._controller: BaseController = PIController(sync_cfg.get("pi", {}) or {})
        elif ctrl_name == "kalman":
            from mesh.kalman_controller import KalmanController
            self._controller = KalmanController(sync_cfg.get("kalman", {}) or {})
        else:
            raise ValueError(f"[{self.node_id}] unknown controller '{ctrl_name}'")

        # Adaptive R (Kalman measurement noise) config
        ar = (sync_cfg.get("kalman", {}) or {}).get("adaptive_r", {}) or {}
        self._R_min = float(ar.get("R_min", 2.5e-9))          # seconds^2
        self._c_rtt = float(ar.get("c_rtt", 0.5))

        # limits / timing
        self._max_slew_per_second_ms = float(sync_cfg.get("max_slew_per_second_ms", 10.0))

        # ROBUST: Timeout config (aus Forum)
        self._coap_timeout = float(sync_cfg.get("coap_timeout_s", 1.0))
        self._base_interval = float(sync_cfg.get("beacon_period_s", 0.5))
        self._max_backoff = float(sync_cfg.get("max_backoff_s", 8.0))

        # jitter / weighting
        self._jitter_window = int(sync_cfg.get("jitter_window", 30))
        self._min_samples_for_jitter = int(sync_cfg.get("min_samples_for_jitter", 10))
        self._max_weight = float(sync_cfg.get("max_weight", 1e6))

        self._drift_damping = float(sync_cfg.get("drift_damping", 0.0))
        self._min_samples_before_log = int(sync_cfg.get("min_samples_before_log", 15))

        self._peer: Dict[str, PeerStats] = {}
        self._last_global_update_mono: Optional[float] = None

        # Last controller action (for dashboard / logging)
        self._last_control_debug: Dict[str, Any] = {
            "dt_s": None,
            "delta_desired_ms": None,
            "delta_applied_ms": None,
            "slew_clipped": None,
            "t_wall": None,
        }

        # ROBUST: Measurement queue (MUST be created in running loop on Py3.7)
        self._measurement_queue: Optional[asyncio.Queue] = None
        self._worker_tasks: List[asyncio.Task] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def mesh_time(self) -> float:
        """
        Monotonic mesh time.

        Uses local monotonic clock + boot_epoch to get a wall-like epoch,
        then applies the mesh offset. This stays strictly monotonic even if
        system wall time jumps (NTP adjustments, manual changes, DST, etc.).
        """
        return time.monotonic() + self._boot_epoch + self._offset

    def get_offset(self) -> float:
        return self._offset

    def last_control_debug(self) -> Dict[str, Any]:
        return dict(self._last_control_debug or {})


    def is_warmed_up(self) -> bool:
        return self._is_root or self._beacon_count >= self._min_beacons_for_warmup

    # -----------------------------------------------------------------
    # Robust stats / weights
    # -----------------------------------------------------------------

    @staticmethod
    def _quantile_sorted(xs: List[float], q: float) -> float:
        n = len(xs)
        if n == 0:
            return 0.0
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        return xs[lo] * (1.0 - (pos - lo)) + xs[hi] * (pos - lo)

    def _compute_iqr(self, xs: List[float]) -> float:
        if len(xs) < 2:
            return 0.0
        xs = sorted(xs)
        return self._quantile_sorted(xs, 0.75) - self._quantile_sorted(xs, 0.25)

    def _update_link_jitter(self, peer_id: str, rtt_s: float) -> None:
        st = self._peer.setdefault(peer_id, PeerStats())
        st.rtt_samples.append(rtt_s)
        st.rtt_last = rtt_s

        if len(st.rtt_samples) > self._jitter_window:
            st.rtt_samples.pop(0)

        if len(st.rtt_samples) >= self._min_samples_for_jitter:
            iqr = self._compute_iqr(st.rtt_samples)
            st.sigma = max(0.7413 * iqr, 1e-6)

            # Robust baseline RTT (median of window)
            xs_sorted = sorted(st.rtt_samples)
            st.rtt_baseline = self._quantile_sorted(xs_sorted, 0.50)


    def _weight_for_peer(self, peer_id: str) -> float:
        st = self._peer.get(peer_id)
        if not st or st.sigma is None:
            return 1.0
        return min(1.0 / (st.sigma * st.sigma), self._max_weight)

    def _noise_for_peer(self, peer_id: str, rtt_s: float) -> float:
        """
        Adaptive measurement variance R (seconds^2) for Kalman.
        Uses:
          - sigma_rtt from IQR (PeerStats.sigma)
          - rtt baseline (PeerStats.rtt_baseline)
          - rtt excess penalty (queueing proxy)
        """
        st = self._peer.get(peer_id)
        if st is None:
            return 1e-6

        sigma_rtt = float(st.sigma) if st.sigma is not None else 0.0
        R = (sigma_rtt * sigma_rtt) / 4.0

        rtt0 = st.rtt_baseline if st.rtt_baseline is not None else (st.rtt_last or rtt_s)
        excess = max(0.0, float(rtt_s) - float(rtt0))
        R += (self._c_rtt * excess) ** 2

        return max(self._R_min, R)

    # -----------------------------------------------------------------
    # Bootstrap
    # -----------------------------------------------------------------

    def _bootstrap_allowed(self, peer_id: str, theta_s: float) -> bool:
        if self._is_root or self._bootstrapped:
            return False
        if self._bootstrap_peers and peer_id not in self._bootstrap_peers:
            return False
        if self._bootstrap_theta_max_s > 0.0 and abs(theta_s) > self._bootstrap_theta_max_s:
            return False
        return True

    def _try_bootstrap(self, peer_id: str, peer_offset_s: float, theta_s: float) -> bool:
        if not self._bootstrap_allowed(peer_id, theta_s):
            return False

        self._offset = peer_offset_s - theta_s
        self._bootstrapped = True
        self._controller.on_bootstrap()

        log.info("[%s] BOOTSTRAP via %s: offset=%.3f ms", self.node_id, peer_id, self._offset * 1000)
        return True

    # -----------------------------------------------------------------
    # Control application
    # -----------------------------------------------------------------

    def _compute_dt(self) -> float:
        now = time.monotonic()
        last = self._last_global_update_mono or now
        self._last_global_update_mono = now
        return max(now - last, 1e-3)

    def _global_slew_clip(self, delta_s: float, dt_s: float) -> float:
        if self._max_slew_per_second_ms <= 0:
            return delta_s
        max_step = (self._max_slew_per_second_ms / 1000.0) * dt_s
        return max(-max_step, min(max_step, delta_s))

    def _apply_global_control(self, meas: List[Measurement]) -> None:
        dt = self._compute_dt()

        if self._controller.__class__.__name__ == "KalmanController":
            delta_desired = self._controller.compute_delta(
                self._offset, meas, self._weight_for_peer, dt, noise_fn=self._noise_for_peer
            )
        else:
            delta_desired = self._controller.compute_delta(
                self._offset, meas, self._weight_for_peer, dt
            )


        delta_applied = self._global_slew_clip(delta_desired, dt)

        # --- controller debug (wanted vs applied) ---
        delta_desired_ms = float(delta_desired * 1000.0)
        delta_applied_ms = float(delta_applied * 1000.0)
        slew_clipped = (abs(delta_applied - delta_desired) > 1e-12)

        self._last_control_debug = {
            "dt_s": float(dt),
            "delta_desired_ms": delta_desired_ms,
            "delta_applied_ms": delta_applied_ms,
            "slew_clipped": bool(slew_clipped),
            "t_wall": float(time.time()),
        }

        self._offset += delta_applied

        #CRITICAL: tell controller what actually happened
        if hasattr(self._controller, "commit_applied_offset"):
            self._controller.commit_applied_offset(self._offset)

        if self._drift_damping > 0.0:
            self._offset *= max(0.0, 1.0 - self._drift_damping * dt)

    # -----------------------------------------------------------------
    # ROBUST: Link Metrics Logging (mit Timeout)
    # -----------------------------------------------------------------

    async def _log_link_metrics(
            self,
            peer_id: str,
            rtt_s: float,
            theta_s: float,
            client_ctx: aiocoap.Context
    ) -> None:
        """Loggt Link-Metriken mit TIMEOUT."""
        st = self._peer.get(peer_id)
        if st is None or st.good_samples < self._min_samples_before_log:
            return

        theta_ms = theta_s * 1000.0
        rtt_ms = rtt_s * 1000.0
        sigma_ms = st.sigma * 1000.0 if st.sigma is not None else None

        t_wall = time.time()
        t_mono = time.monotonic()
        t_mesh = self.mesh_time()

        # Option 1: Lokales Logging
        if self._storage is not None:
            try:
                dbg = self.last_control_debug()
                self._storage.insert_ntp_reference(
                    node_id=self.node_id,
                    t_wall=t_wall,
                    t_mono=t_mono,
                    t_mesh=t_mesh,
                    offset=self._offset,
                    err_mesh_vs_wall=t_mesh - t_wall,
                    peer_id=peer_id,
                    theta_ms=theta_ms,
                    rtt_ms=rtt_ms,
                    sigma_ms=sigma_ms,
                    # NEW: controller debug
                    delta_desired_ms=dbg.get("delta_desired_ms"),
                    delta_applied_ms=dbg.get("delta_applied_ms"),
                    dt_s=dbg.get("dt_s"),
                    slew_clipped=dbg.get("slew_clipped"),
                )

            except Exception as e:
                log.error("[%s] DB logging failed for %s: %s", self.node_id, peer_id, e)

        # Option 2: Telemetrie (MIT TIMEOUT!)
        elif self._telemetry_sink_ip is not None:
            try:
                uri = f"coap://{self._telemetry_sink_ip}/relay/ingest/link"

                payload = {
                    "node_id": self.node_id,
                    "peer_id": peer_id,
                    "theta_ms": theta_ms,
                    "rtt_ms": rtt_ms,
                    "sigma_ms": sigma_ms,
                    "t_wall": t_wall,
                    "t_mono": t_mono,
                    "t_mesh": t_mesh,
                    "offset": self._offset,
                }

                dbg = self.last_control_debug()
                payload.update({
                    "delta_desired_ms": dbg.get("delta_desired_ms"),
                    "delta_applied_ms": dbg.get("delta_applied_ms"),
                    "dt_s": dbg.get("dt_s"),
                    "slew_clipped": dbg.get("slew_clipped"),
                })


                req = aiocoap.Message(
                    code=aiocoap.POST,
                    uri=uri,
                    payload=json.dumps(payload).encode("utf-8"),
                )

                # ROBUST: Hard timeout!
                await asyncio.wait_for(
                    client_ctx.request(req).response,
                    timeout=self._coap_timeout
                )

            except asyncio.TimeoutError:
                log.warning("[%s] Telemetry timeout for %s", self.node_id, peer_id)
            except Exception as e:
                log.warning("[%s] Telemetry failed for %s: %s", self.node_id, peer_id, type(e).__name__)

    # -----------------------------------------------------------------
    # ROBUST: Per-Peer Worker (aus Forum)
    # -----------------------------------------------------------------

    async def _peer_worker(
            self,
            peer_id: str,
            peer_ip: str,
            client_ctx: aiocoap.Context
    ) -> None:
        """
        ROBUST peer worker mit Backoff und Exception Handling.
        Nach Forum-Best-Practices.
        """
        backoff = self._base_interval
        st = self._peer.setdefault(peer_id, PeerStats())
        st.state = "DOWN"

        log.info("[%s] Starting peer worker for %s (%s)", self.node_id, peer_id, peer_ip)

        while True:
            uri = f"coap://{peer_ip}/sync/beacon"

            t1_m = time.monotonic()
            payload = {
                "src": self.node_id,
                "dst": peer_id,
                "t1": t1_m,
                "boot_epoch": self._boot_epoch,
                "offset": self._offset,
            }

            req = aiocoap.Message(
                code=aiocoap.POST,
                uri=uri,
                payload=json.dumps(payload).encode()
            )

            try:
                # ROBUST: Hard timeout!
                resp = await asyncio.wait_for(
                    client_ctx.request(req).response,
                    timeout=self._coap_timeout
                )

                t4_m = time.monotonic()

                data = json.loads(resp.payload.decode())
                t2_m = float(data["t2"])
                t3_m = float(data["t3"])
                peer_offset = float(data["offset"])
                peer_boot_epoch = float(data.get("boot_epoch"))

                t1 = t1_m + self._boot_epoch
                t4 = t4_m + self._boot_epoch
                t2 = t2_m + peer_boot_epoch
                t3 = t3_m + peer_boot_epoch

                rtt = (t4 - t1) - (t3 - t2)
                theta = ((t2 - t1) + (t3 - t4)) / 2.0

                # SUCCESS!
                if st.state != "UP":
                    log.info("[%s] Peer %s is UP", self.node_id, peer_id)
                    st.state = "UP"

                self._beacon_count += 1
                st.good_samples += 1
                st.theta_last = theta
                st.last_theta_epoch = time.time()
                self._update_link_jitter(peer_id, rtt)

                # Bootstrap
                did_bs = self._try_bootstrap(peer_id, peer_offset, theta)

                # Queue measurement for control
                if not did_bs:
                    q = self._measurement_queue
                    if q is not None:
                        await q.put((peer_id, rtt, theta, peer_offset))

                # Log metrics
                await self._log_link_metrics(peer_id, rtt, theta, client_ctx)

                # Reset backoff
                backoff = self._base_interval
                await asyncio.sleep(self._base_interval)

            except asyncio.TimeoutError:
                # EXPECTED wenn Peer down
                if st.state != "DOWN":
                    log.warning("[%s] Peer %s timeout", self.node_id, peer_id)
                    st.state = "DOWN"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)

            except Exception as e:
                # Network error, parse error, etc.
                if st.state != "DOWN":
                    log.warning("[%s] Peer %s error: %s", self.node_id, peer_id, type(e).__name__)
                    st.state = "DOWN"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._max_backoff)

    # -----------------------------------------------------------------
    # ROBUST: Control Loop (verarbeitet Measurements)
    # -----------------------------------------------------------------

    async def _control_loop(self) -> None:
        """
        Sammelt Measurements von allen Peers und wendet Control an.
        """
        log.info("[%s] Starting control loop", self.node_id)

        measurement_buffer: List[Measurement] = []
        last_control = time.monotonic()

        while True:
            try:
                # Sammle Measurements für control_interval
                timeout = max(0.1, self._base_interval - (time.monotonic() - last_control))

                try:
                    q = self._measurement_queue
                    if q is None:
                        await asyncio.sleep(0.2)
                        continue

                    meas = await asyncio.wait_for(q.get(), timeout=timeout)
                    measurement_buffer.append(meas)

                except asyncio.TimeoutError:
                    pass

                # Apply control wenn genug Zeit vergangen
                now = time.monotonic()
                if now - last_control >= self._base_interval:
                    if measurement_buffer:
                        self._apply_global_control(measurement_buffer)
                        measurement_buffer.clear()
                    last_control = now

            except (asyncio.CancelledError, concurrent.futures.CancelledError):
                log.info("[%s] Control loop cancelled", self.node_id)
                return

            except Exception as e:
                log.exception("[%s] Control loop error: %s", self.node_id, e)
                await asyncio.sleep(1.0)
    # -----------------------------------------------------------------
    # ROBUST: Start/Stop
    # -----------------------------------------------------------------

    async def start(self, client_ctx: aiocoap.Context) -> None:
        """
        Startet alle Peer-Worker und Control-Loop.
        ROBUST: Separate Tasks, kein Blocking.
        """
        if IS_WINDOWS or self._is_root:
            log.info("[%s] Skipping beacon workers (Windows or Root)", self.node_id)
            return

        # ---- FIX (Py3.7): create asyncio primitives in the running loop ----
        self._loop = asyncio.get_running_loop()
        if self._measurement_queue is None:
            self._measurement_queue = asyncio.Queue()
        # -------------------------------------------------------------------

        # Start control loop
        self._worker_tasks.append(
            spawn_task(self._control_loop(), f"{self.node_id}:control")
        )

        # Start peer workers
        for peer_id in self.neighbors:
            peer_ip = self.neighbor_ips.get(peer_id)
            if not peer_ip:
                log.warning("[%s] No IP for peer %s", self.node_id, peer_id)
                continue

            self._worker_tasks.append(
                spawn_task(
                    self._peer_worker(peer_id, peer_ip, client_ctx),
                    f"{self.node_id}:peer:{peer_id}"
                )
            )

        log.info("[%s] Started %d workers", self.node_id, len(self._worker_tasks))

    async def stop(self) -> None:
        """Stoppt alle Worker."""
        log.info("[%s] Stopping %d workers", self.node_id, len(self._worker_tasks))

        for task in self._worker_tasks:
            task.cancel()

        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        self._measurement_queue = None
        self._loop = None
