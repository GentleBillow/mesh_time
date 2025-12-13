# -*- coding: utf-8 -*-
# mesh/sync.py - COMPLETE FIX mit Link-Metrics Telemetrie

from __future__ import annotations

import json
import math
import platform
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable

import aiocoap

IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class PeerStats:
    """Per-peer rolling link statistics (seconds)."""
    rtt_samples: List[float] = field(default_factory=list)
    sigma: Optional[float] = None
    theta_last: Optional[float] = None  # theta ≈ o_peer - o_self
    good_samples: int = 0
    last_theta_epoch: Optional[float] = None
    # NEU: Letzte RTT für Telemetrie
    rtt_last: Optional[float] = None


# ---------------------------------------------------------------------
# Controller interface + implementations
# ---------------------------------------------------------------------

Measurement = Tuple[str, float, float, float]
# (peer_id, rtt_s, theta_s, peer_offset_s)

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
    """
    Weighted global PI controller.

    error_i = o_self - (o_peer - theta_i)
    e = Σ(w_i * error_i) / Σ(w_i)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.kp = float(cfg.get("kp", 0.02))
        self.ki = float(cfg.get("ki", 0.001))
        self.i_leak = float(cfg.get("i_leak", 0.0))  # 1/s
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

        # leaky integrator
        if self.i_leak > 0.0:
            self.i_state *= max(0.0, 1.0 - self.i_leak * dt_s)

        self.i_state += e * dt_s
        if self.i_max > 0.0:
            self.i_state = max(-self.i_max, min(self.i_max, self.i_state))

        return -self.kp * e - self.ki * self.i_state

    def snapshot(self) -> Dict[str, Any]:
        return {
            "type": "pi",
            "kp": self.kp,
            "ki": self.ki,
            "i_state_ms": self.i_state * 1000.0,
            "i_max_ms": self.i_max * 1000.0,
            "i_leak": self.i_leak,
        }


# ---------------------------------------------------------------------
# Sync module
# ---------------------------------------------------------------------

class SyncModule:
    """
    Mesh time synchronization via CoAP beacons + NTP 4-timestamp estimate.
    Controller is pluggable (PI / Kalman).

    FIX: Logs link metrics (theta, rtt, sigma) either locally OR via telemetry callback.
    """

    def __init__(
            self,
            node_id: str,
            neighbors: List[str],
            neighbor_ips: Dict[str, str],
            sync_cfg: Optional[Dict[str, Any]] = None,
            storage=None,
            telemetry_callback=None,  # NEU: Callback für Telemetrie
    ) -> None:
        sync_cfg = sync_cfg or {}

        self.node_id = node_id
        self.neighbors = list(neighbors)
        self.neighbor_ips = dict(neighbor_ips)
        self._storage = storage
        self._telemetry_callback = telemetry_callback  # NEU

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

        # limits / timing
        self._max_slew_per_second_ms = float(sync_cfg.get("max_slew_per_second_ms", 10.0))
        self._coap_timeout = float(sync_cfg.get("coap_timeout_s", 0.5))

        # jitter / weighting
        self._jitter_window = int(sync_cfg.get("jitter_window", 30))
        self._min_samples_for_jitter = int(sync_cfg.get("min_samples_for_jitter", 10))
        self._max_weight = float(sync_cfg.get("max_weight", 1e6))

        self._drift_damping = float(sync_cfg.get("drift_damping", 0.0))
        self._min_samples_before_log = int(sync_cfg.get("min_samples_before_log", 15))

        self._peer: Dict[str, PeerStats] = {}
        self._last_global_update_mono: Optional[float] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def mesh_time(self) -> float:
        return time.time() + self._offset

    def get_offset(self) -> float:
        return self._offset

    def is_warmed_up(self) -> bool:
        return self._is_root or self._beacon_count >= self._min_beacons_for_warmup

    def get_peer_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        NEU: Gibt aktuelle Peer-Statistiken zurück für Telemetrie.
        Returns: {peer_id: {theta_ms, rtt_ms, sigma_ms, samples}}
        """
        result = {}
        for peer_id, st in self._peer.items():
            if st.good_samples < self._min_samples_before_log:
                continue

            result[peer_id] = {
                "theta_ms": st.theta_last * 1000.0 if st.theta_last is not None else None,
                "rtt_ms": st.rtt_last * 1000.0 if st.rtt_last is not None else None,
                "sigma_ms": st.sigma * 1000.0 if st.sigma is not None else None,
                "samples": st.good_samples,
            }
        return result

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
        st.rtt_last = rtt_s  # NEU: Speichere letzte RTT

        if len(st.rtt_samples) > self._jitter_window:
            st.rtt_samples.pop(0)

        if len(st.rtt_samples) >= self._min_samples_for_jitter:
            iqr = self._compute_iqr(st.rtt_samples)
            st.sigma = max(0.7413 * iqr, 1e-6)

    def _weight_for_peer(self, peer_id: str) -> float:
        st = self._peer.get(peer_id)
        if not st or st.sigma is None:
            return 1.0
        return min(1.0 / (st.sigma * st.sigma), self._max_weight)

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

        print(f"[{self.node_id}] BOOTSTRAP via {peer_id}: offset={self._offset * 1000:.3f} ms")
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
        delta = self._controller.compute_delta(self._offset, meas, self._weight_for_peer, dt)
        delta = self._global_slew_clip(delta, dt)
        self._offset += delta

        if self._drift_damping > 0.0:
            self._offset *= max(0.0, 1.0 - self._drift_damping * dt)

        print(f"[{self.node_id}] control: Δ={delta * 1000:.3f} ms, offset={self._offset * 1000:.3f} ms")

    # -----------------------------------------------------------------
    # FIX: Link Metrics Logging/Telemetry
    # -----------------------------------------------------------------

    def _log_link_metrics(self, peer_id: str, rtt_s: float, theta_s: float) -> None:
        """
        Loggt Link-Metriken entweder:
        1. Lokal in DB (wenn self._storage vorhanden)
        2. Via Telemetrie-Callback (wenn kein Storage aber Callback vorhanden)
        """
        st = self._peer.get(peer_id)
        if st is None or st.good_samples < self._min_samples_before_log:
            return

        theta_ms = theta_s * 1000.0
        rtt_ms = rtt_s * 1000.0
        sigma_ms = st.sigma * 1000.0 if st.sigma is not None else None

        t_wall = time.time()
        t_mono = time.monotonic()
        t_mesh = self.mesh_time()

        # Option 1: Lokales Logging (nur Node C)
        if self._storage is not None:
            try:
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
                )
                # Optional: Verbose logging
                # print(f"[{self.node_id}] Link to {peer_id}: θ={theta_ms:.2f}ms, RTT={rtt_ms:.2f}ms, σ={sigma_ms:.2f}ms")
            except Exception as e:
                print(f"[{self.node_id}] Link metrics DB logging failed for peer {peer_id}: {e}")

        # Option 2: Telemetrie (Nodes A, B, D)
        elif self._telemetry_callback is not None:
            try:
                self._telemetry_callback(peer_id, theta_ms, rtt_ms, sigma_ms)
            except Exception as e:
                print(f"[{self.node_id}] Link metrics telemetry failed for peer {peer_id}: {e}")

    # -----------------------------------------------------------------
    # Beacon roundtrip
    # -----------------------------------------------------------------

    async def send_beacons(self, client_ctx: aiocoap.Context) -> None:
        # DEBUG: Am Anfang der Funktion
        print(f"[DEBUG] {self.node_id}: send_beacons called, is_root={self._is_root}, IS_WINDOWS={IS_WINDOWS}")

        if IS_WINDOWS or self._is_root:
            return

        meas: List[Measurement] = []

        for peer_id in self.neighbors:
            ip = self.neighbor_ips.get(peer_id)
            if not ip:
                continue

            uri = f"coap://{ip}/sync/beacon"

            t1_m = time.monotonic()
            payload = {
                "src": self.node_id,
                "dst": peer_id,
                "t1": t1_m,
                "boot_epoch": self._boot_epoch,
                "offset": self._offset,
            }

            req = aiocoap.Message(code=aiocoap.POST, uri=uri, payload=json.dumps(payload).encode())
            try:
                resp = await client_ctx.request(req).response
            except Exception:
                continue

            t4_m = time.monotonic()

            try:
                data = json.loads(resp.payload.decode())
                t2_m = float(data["t2"])
                t3_m = float(data["t3"])
                peer_offset = float(data["offset"])
                peer_boot_epoch = float(data.get("boot_epoch"))
            except Exception:
                continue

            t1 = t1_m + self._boot_epoch
            t4 = t4_m + self._boot_epoch
            t2 = t2_m + peer_boot_epoch
            t3 = t3_m + peer_boot_epoch

            rtt = (t4 - t1) - (t3 - t2)
            theta = ((t2 - t1) + (t3 - t4)) / 2.0

            self._beacon_count += 1
            st = self._peer.setdefault(peer_id, PeerStats())
            st.good_samples += 1
            st.theta_last = theta
            st.last_theta_epoch = time.time()
            self._update_link_jitter(peer_id, rtt)

            did_bs = self._try_bootstrap(peer_id, peer_offset, theta)

            if not did_bs:
                meas.append((peer_id, rtt, theta, peer_offset))

            # FIX: Logge Link-Metriken nach jedem erfolgreichen Beacon!
            self._log_link_metrics(peer_id, rtt, theta)

        if meas:
            self._apply_global_control(meas)