# -*- coding: utf-8 -*-
# mesh/sync.py

from __future__ import annotations

import asyncio
import json
import math
import platform
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import aiocoap

IS_WINDOWS = platform.system() == "Windows"


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------

@dataclass
class PeerStats:
    """Per-peer rolling link statistics (seconds)."""
    rtt_samples: List[float] = field(default_factory=list)
    sigma: Optional[float] = None                 # robust jitter proxy (s)
    theta_last: Optional[float] = None            # last theta estimate (s), theta ≈ peer_clock - self_clock
    good_samples: int = 0

    last_update_mono: Optional[float] = None      # for per-peer bookkeeping
    last_theta_epoch: Optional[float] = None      # time.time() when theta_last was computed


# ---------------------------------------------------------------------
# Sync module
# ---------------------------------------------------------------------

class SyncModule:
    """
    Mesh time synchronization via CoAP beacons + NTP 4-timestamp estimate.

    State (per node)
    ----------------
      offset o (seconds), such that:
          mesh_time() = time.time() + o

    Measurement (per link)
    ----------------------
      theta ≈ peer_clock - self_clock ≈ o_peer - o_self

    Target (for our offset)
    -----------------------
      o_self ≈ o_peer - theta

    Controller
    ----------
      Global (aggregated) PI controller on the offset error:
          error_i = o_self - (o_peer - theta_i)

      Weighted average error:
          e = Σ(w_i * error_i) / Σ(w_i)

      PI:
          I <- clamp( leak(I) + e*dt, ±I_max )
          delta = -Kp * e - Ki * I

      Then global slew clipping (ms/s) and apply to offset.

    Notes
    -----
    - We "lift" monotonic timestamps into an epoch-like scale by exchanging boot_epoch.
    - This keeps monotonic stability while allowing cross-device NTP math.
    """

    def __init__(
        self,
        node_id: str,
        neighbors: List[str],
        neighbor_ips: Dict[str, str],
        sync_cfg: Optional[Dict] = None,
        storage=None,
    ) -> None:
        sync_cfg = sync_cfg or {}

        self.node_id = node_id
        self.neighbors = list(neighbors)
        self.neighbor_ips = dict(neighbor_ips)
        self._storage = storage

        # Role
        self._is_root = bool(sync_cfg.get("is_root", False))

        # Bootstrap / warmup
        self._bootstrapped = bool(self._is_root)
        self._beacon_count = 0
        self._min_beacons_for_warmup = int(sync_cfg.get("min_beacons_for_warmup", 15))
        self._bootstrap_theta_max_s = float(sync_cfg.get("bootstrap_theta_max_s", 0.0))
        self._bootstrap_peers = set(sync_cfg.get("bootstrap_peers", []) or [])

        # Time lifting
        self._boot_epoch = time.time() - time.monotonic()

        # Offset init
        initial_offset_ms = float(sync_cfg.get("initial_offset_ms", 200.0))
        if self._is_root:
            self._offset = 0.0
        else:
            self._offset = random.uniform(-initial_offset_ms / 1000.0, +initial_offset_ms / 1000.0)

        # Controller (PI) — single source of truth
        self._kp = float(sync_cfg.get("kp", 0.02))                 # proportional gain
        self._ki = float(sync_cfg.get("ki", 0.001))                # integral gain (1/s)
        self._i_state = 0.0                                        # integral state (s)
        self._i_max = float(sync_cfg.get("i_max_ms", 200.0)) / 1000.0  # anti-windup clamp (s)
        self._i_leak = float(sync_cfg.get("i_leak", 0.0))          # optional leak (1/s)

        # Slew limit (global) in ms/s
        self._max_slew_per_second_ms = float(sync_cfg.get("max_slew_per_second_ms", 10.0))

        # CoAP timeout
        self._coap_timeout = float(sync_cfg.get("coap_timeout_s", 0.5))

        # Jitter estimation (RTT-based)
        self._jitter_window = int(sync_cfg.get("jitter_window", 30))
        self._min_samples_for_jitter = int(sync_cfg.get("min_samples_for_jitter", 10))
        self._max_weight = float(sync_cfg.get("max_weight", 1e6))

        # Optional gentle pull-to-zero (NOT a controller; just damping)
        self._drift_damping = float(sync_cfg.get("drift_damping", 0.0))

        # Logging gating
        self._min_samples_before_log = int(sync_cfg.get("min_samples_before_log", 15))

        # Peer stats
        self._peer: Dict[str, PeerStats] = {}

        # Global update timing
        self._last_global_update_mono: Optional[float] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def mesh_time(self) -> float:
        """Epoch-like mesh time = wallclock + offset."""
        return time.time() + self._offset

    def get_offset(self) -> float:
        return self._offset

    def is_warmed_up(self) -> bool:
        if self._is_root:
            return True
        return self._beacon_count >= self._min_beacons_for_warmup

    def status_snapshot(self) -> dict:
        peer_theta = {}
        peer_sigma = {}
        peer_rtt = {}
        peer_good = {}

        for pid, st in self._peer.items():
            if st.theta_last is not None:
                peer_theta[pid] = st.theta_last
            if st.sigma is not None:
                peer_sigma[pid] = st.sigma
            if st.rtt_samples:
                peer_rtt[pid] = st.rtt_samples[-1]
            peer_good[pid] = st.good_samples

        return {
            "node_id": self.node_id,
            "is_root": self._is_root,
            "bootstrapped": self._bootstrapped,
            "mesh_time": self.mesh_time(),
            "offset_estimate": self._offset,
            "offset_estimate_ms": self._offset * 1000.0,
            "neighbors": list(self.neighbors),
            "peer_theta": peer_theta,
            "peer_theta_ms": {k: v * 1000.0 for k, v in peer_theta.items()},
            "peer_sigma": peer_sigma,
            "peer_sigma_ms": {k: v * 1000.0 for k, v in peer_sigma.items()},
            "peer_rtt_s": peer_rtt,
            "peer_rtt_ms": {k: v * 1000.0 for k, v in peer_rtt.items()},
            "peer_good_samples": peer_good,
            "sync_config": {
                "kp": self._kp,
                "ki": self._ki,
                "i_max_ms": self._i_max * 1000.0,
                "i_leak": self._i_leak,
                "max_slew_per_second_ms": self._max_slew_per_second_ms,
                "coap_timeout_s": self._coap_timeout,
                "jitter_window": self._jitter_window,
                "min_samples_for_jitter": self._min_samples_for_jitter,
                "min_samples_before_log": self._min_samples_before_log,
                "bootstrap_theta_max_s": self._bootstrap_theta_max_s,
            },
        }

    def inject_disturbance(self, delta_s: float) -> None:
        self._offset += float(delta_s)
        print(f"[{self.node_id}] inject_disturbance: delta={delta_s*1000.0:.1f} ms, new_offset={self._offset*1000.0:.1f} ms")

    # -----------------------------------------------------------------
    # Robust stats / weights
    # -----------------------------------------------------------------

    @staticmethod
    def _quantile_sorted(xs: List[float], q: float) -> float:
        n = len(xs)
        if n == 0:
            return 0.0
        if n == 1:
            return xs[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return xs[lo]
        frac = pos - lo
        return xs[lo] * (1.0 - frac) + xs[hi] * frac

    @classmethod
    def _compute_iqr(cls, samples: List[float]) -> float:
        if len(samples) < 2:
            return 0.0
        xs = sorted(samples)
        q1 = cls._quantile_sorted(xs, 0.25)
        q3 = cls._quantile_sorted(xs, 0.75)
        return float(q3 - q1)

    def _update_link_jitter(self, peer_id: str, rtt_s: float) -> None:
        st = self._peer.setdefault(peer_id, PeerStats())
        st.rtt_samples.append(rtt_s)
        if len(st.rtt_samples) > self._jitter_window:
            st.rtt_samples.pop(0)

        if len(st.rtt_samples) >= self._min_samples_for_jitter:
            iqr = self._compute_iqr(st.rtt_samples)
            st.sigma = max(0.7413 * iqr, 1e-6)

    def _weight_for_peer(self, peer_id: str) -> float:
        st = self._peer.get(peer_id)
        if not st or st.sigma is None:
            return 1.0
        w = 1.0 / (st.sigma * st.sigma)
        return min(w, self._max_weight)

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
        """
        theta ≈ o_peer - o_self  =>  o_self ≈ o_peer - theta
        """
        if not self._bootstrap_allowed(peer_id, theta_s):
            return False

        old = self._offset
        self._offset = peer_offset_s - theta_s
        self._bootstrapped = True
        self._i_state = 0.0  # reset integrator on bootstrap (safer)

        print(
            f"[{self.node_id}] BOOTSTRAP with {peer_id}: "
            f"theta={theta_s*1000.0:.3f} ms, old_offset={old*1000.0:.3f} ms, new_offset={self._offset*1000.0:.3f} ms"
        )
        return True

    # -----------------------------------------------------------------
    # Apply update (PI + slew + optional damping)
    # -----------------------------------------------------------------

    def _apply_delta(self, delta_s: float, dt_s: float) -> None:
        self._offset += delta_s

        if self._drift_damping > 0.0:
            self._offset *= max(0.0, 1.0 - self._drift_damping * dt_s)

    def _compute_dt(self) -> float:
        now_m = time.monotonic()
        last_m = self._last_global_update_mono if self._last_global_update_mono is not None else now_m
        dt = max(now_m - last_m, 1e-3)
        self._last_global_update_mono = now_m
        return dt

    def _global_slew_clip(self, delta_s: float, dt_s: float) -> float:
        if self._max_slew_per_second_ms <= 0.0:
            return delta_s
        max_step = (self._max_slew_per_second_ms / 1000.0) * dt_s
        return max(-max_step, min(max_step, delta_s))

    def _pi_update_from_errors(self, e_s: float, dt_s: float) -> float:
        # optional leaky integrator
        if self._i_leak > 0.0:
            self._i_state *= max(0.0, 1.0 - self._i_leak * dt_s)

        self._i_state += e_s * dt_s
        if self._i_max > 0.0:
            self._i_state = max(-self._i_max, min(self._i_max, self._i_state))

        delta = -self._kp * e_s - self._ki * self._i_state
        return delta

    # -----------------------------------------------------------------
    # Logging gating
    # -----------------------------------------------------------------

    def _should_log(self, peer_id: str) -> bool:
        if self._storage is None:
            return False
        st = self._peer.get(peer_id)
        if not st:
            return False
        return self._bootstrapped and (st.good_samples >= self._min_samples_before_log)

    def _log_sync_sample(
        self,
        peer: str,
        rtt_s: float,
        theta_s: float,
    ) -> None:
        if not self._should_log(peer):
            return

        t_wall = time.time()
        t_mono = time.monotonic()
        t_mesh = self.mesh_time()
        offset = self._offset
        err_mesh_vs_wall = t_mesh - t_wall

        st = self._peer.get(peer)
        sigma = st.sigma if st and st.sigma is not None else None

        try:
            self._storage.insert_ntp_reference(
                node_id=self.node_id,
                t_wall=t_wall,
                t_mono=t_mono,
                t_mesh=t_mesh,
                offset=offset,
                err_mesh_vs_wall=err_mesh_vs_wall,
                peer_id=peer,
                theta_ms=theta_s * 1000.0,
                rtt_ms=rtt_s * 1000.0,
                sigma_ms=(sigma * 1000.0) if sigma is not None else None,
            )
        except Exception as e:
            print(f"[{self.node_id}] _log_sync_sample failed: {e}")

    # -----------------------------------------------------------------
    # Beacon roundtrip
    # -----------------------------------------------------------------

    async def send_beacons(self, client_ctx: aiocoap.Context) -> None:
        """
        For each neighbor:
          - do NTP 4-ts estimate -> theta, rtt
          - update jitter sigma from rtt
          - optionally bootstrap
        After collecting measurements:
          - compute weighted mean error e
          - do ONE PI update and apply (with global slew clip)
        """
        if IS_WINDOWS or self._is_root:
            return

        # Collect per-peer measurements for one global update
        meas: List[Tuple[str, float, float, float]] = []
        # (peer_id, rtt_s, theta_s, peer_offset_s)

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

            req = aiocoap.Message(
                code=aiocoap.POST,
                uri=uri,
                payload=json.dumps(payload).encode("utf-8"),
            )

            try:
                resp = await asyncio.wait_for(client_ctx.request(req).response, timeout=self._coap_timeout)
            except asyncio.TimeoutError:
                print(f"[{self.node_id}] beacon to {peer_id} timed out after {self._coap_timeout:.3f}s")
                continue
            except Exception as e:
                print(f"[{self.node_id}] beacon to {peer_id} failed: {e}")
                continue

            t4_m = time.monotonic()

            try:
                data = json.loads(resp.payload.decode("utf-8"))
                t2_m = float(data["t2"])
                t3_m = float(data["t3"])
                peer_offset = float(data["offset"])
                peer_boot_epoch = data.get("boot_epoch", None)
                peer_boot_epoch = float(peer_boot_epoch) if peer_boot_epoch is not None else None
            except Exception as e:
                print(f"[{self.node_id}] invalid beacon reply from {peer_id}: {e}")
                continue

            # Lift to epoch-like scale when possible
            if peer_boot_epoch is not None:
                t1 = t1_m + self._boot_epoch
                t4 = t4_m + self._boot_epoch
                t2 = t2_m + peer_boot_epoch
                t3 = t3_m + peer_boot_epoch
            else:
                t1, t4, t2, t3 = t1_m, t4_m, t2_m, t3_m

            rtt = (t4 - t1) - (t3 - t2)
            theta = ((t2 - t1) + (t3 - t4)) / 2.0

            self._beacon_count += 1
            if self._beacon_count == self._min_beacons_for_warmup:
                print(f"[{self.node_id}] sync warmup complete")

            st = self._peer.setdefault(peer_id, PeerStats())
            st.good_samples += 1
            st.theta_last = theta
            st.last_theta_epoch = time.time()
            self._update_link_jitter(peer_id, rtt)

            # Bootstrap (one-time)
            did_bs = self._try_bootstrap(peer_id, peer_offset, theta)

            sigma_ms = (st.sigma * 1000.0) if st.sigma is not None else None
            sigma_str = f"{sigma_ms:.3f} ms" if sigma_ms is not None else "n/a"
            print(
                f"[{self.node_id}] sync with {peer_id}: "
                f"theta={theta*1000.0:.3f} ms, rtt={rtt*1000.0:.3f} ms, sigma={sigma_str}"
            )

            self._log_sync_sample(peer=peer_id, rtt_s=rtt, theta_s=theta)

            if not did_bs:
                meas.append((peer_id, rtt, theta, peer_offset))

        # One global PI update per round
        if meas:
            self._apply_global_pi(meas)

    def _apply_global_pi(self, meas: List[Tuple[str, float, float, float]]) -> None:
        """
        meas: (peer_id, rtt_s, theta_s, peer_offset_s)
        """
        # Weighted mean error
        num_e = 0.0
        den = 0.0

        for peer_id, _rtt, theta_s, peer_offset_s in meas:
            w = self._weight_for_peer(peer_id)
            target = peer_offset_s - theta_s
            error = self._offset - target
            num_e += w * error
            den += w

        if den <= 0.0:
            return

        e = num_e / den

        dt = self._compute_dt()

        # PI controller step
        delta = self._pi_update_from_errors(e, dt)

        # Global slew clip
        delta = self._global_slew_clip(delta, dt)

        self._apply_delta(delta, dt)

        print(
            f"[{self.node_id}] PI aggregate: "
            f"e={e*1000.0:.3f} ms, I={self._i_state*1000.0:.3f} ms, "
            f"delta={delta*1000.0:.3f} ms, offset={self._offset*1000.0:.3f} ms (dt={dt:.3f}s)"
        )
