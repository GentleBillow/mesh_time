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
# Small helpers / data containers
# ---------------------------------------------------------------------

@dataclass
class PeerStats:
    """
    Per-peer rolling link statistics.

    All times are in seconds.
    """
    rtt_samples: List[float] = field(default_factory=list)   # rolling buffer
    sigma: Optional[float] = None                            # jitter std approx (s)
    theta_last: Optional[float] = None                       # last offset estimate (s)
    good_samples: int = 0                                    # count of successful beacons
    last_update_mono: Optional[float] = None                 # monotonic timestamp (local)
    last_theta_epoch: Optional[float] = None                 # epoch time when theta_last computed


# ---------------------------------------------------------------------
# Sync module
# ---------------------------------------------------------------------

class SyncModule:
    """
    Mesh time synchronization using CoAP beacons and an NTP-style 4-timestamp estimate.

    Core concept
    ------------
    Each node i maintains an offset o_i (seconds) such that:

        mesh_time() = local_clock() + o_i

    We estimate the relative offset between peer clocks using the classic NTP formula
    (4 timestamps t1..t4). That gives an estimate:

        theta ≈ (peer_clock - self_clock)  ≈ o_peer - o_self

    Then we update our local offset towards a consensus.

    Key robustness fix: monotonic-to-epoch lifting
    ----------------------------------------------
    Using time.monotonic() directly across machines is WRONG because each machine has
    a different origin (boot time). To make the NTP math meaningful we lift monotonic
    timestamps into a common scale by exchanging:

        boot_epoch = time.time() - time.monotonic()

    Then each node can map local monotonic to an "epoch-like" clock:

        t_epoch = t_mono + boot_epoch

    This keeps the nice monotonic properties for measurement while removing the
    arbitrary boot-time offset between devices.

    Kalman-ready structure
    ----------------------
    The module is structured around:
      - measurement: theta (peer - self)
      - measurement uncertainty proxy: sigma (from RTT IQR)
      - update step: apply correction delta to offset

    Later you can replace the scalar offset update with a Kalman state
    (e.g., [offset, drift]) per node or per link without changing the beacon I/O.
    """

    def __init__(
        self,
        node_id: str,
        neighbors: List[str],
        neighbor_ips: Dict[str, str],
        sync_cfg: Optional[Dict[str, float]] = None,
        storage=None,
    ) -> None:
        self.node_id = node_id
        self.neighbors = list(neighbors)
        self.neighbor_ips = dict(neighbor_ips)
        self._storage = storage

        sync_cfg = sync_cfg or {}

        # Root: server-only (no outbound beacons), keeps offset fixed at 0
        self._is_root = bool(sync_cfg.get("is_root", False))

        # --- algorithm knobs (seconds / ms are noted explicitly) ---
        initial_offset_ms = float(sync_cfg.get("initial_offset_ms", 200.0))
        self._eta = float(sync_cfg.get("eta", 0.02))

        # Slew limit (ms/s in config)
        self._max_slew_per_second_ms = float(sync_cfg.get("max_slew_per_second_ms", 10.0))

        # CoAP timeout
        self._coap_timeout = float(sync_cfg.get("coap_timeout_s", 0.5))

        # Jitter estimation
        self._jitter_window = int(sync_cfg.get("jitter_window", 30))
        self._min_samples_for_jitter = int(sync_cfg.get("min_samples_for_jitter", 10))

        # Logging gating (avoid ugly startup transients in DB/Web-UI)
        self._min_samples_before_log = int(sync_cfg.get("min_samples_before_log", 15))

        # Bootstrap
        # If >0: only bootstrap if abs(theta) <= threshold (seconds)
        self._bootstrap_theta_max_s = float(sync_cfg.get("bootstrap_theta_max_s", 0.0))

        # Prefer bootstrapping from a specific peer (optional, e.g. "C")
        # If set, we only bootstrap when talking to one of these peers.
        self._bootstrap_peers = set(sync_cfg.get("bootstrap_peers", []) or [])

        # Drift damping (optional): small pull towards 0 offset, per second
        self._drift_damping = float(sync_cfg.get("drift_damping", 0.0))

        # Aggregate update (recommended): combine all peer deltas into ONE delta per round
        self._aggregate_updates = bool(sync_cfg.get("aggregate_updates", True))

        # Weight cap to prevent crazy dominance when sigma becomes tiny
        self._max_weight = float(sync_cfg.get("max_weight", 1e6))

        # --- time base lifting ---
        # "epoch at monotonic=0" estimate
        self._boot_epoch = time.time() - time.monotonic()

        # --- state ---
        if self._is_root:
            self._offset = 0.0
            self._bootstrapped = True
        else:
            self._offset = random.uniform(-initial_offset_ms / 1000.0, +initial_offset_ms / 1000.0)
            self._bootstrapped = False

        # Per-peer stats
        self._peer: Dict[str, PeerStats] = {}

        # For overall slew limiting when aggregating
        self._last_global_update_mono: Optional[float] = None

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def mesh_time(self) -> float:
        """Current mesh time in seconds (epoch-like) = time.time() + offset."""
        # We expose mesh time on a wallclock-like scale (epoch seconds).
        # This makes the Web-UI and cross-node comparisons intuitive.
        return time.time() + self._offset

    def get_offset(self) -> float:
        """Current offset in seconds."""
        return self._offset

    def status_snapshot(self) -> dict:
        """Status snapshot for /status endpoint (units: seconds + ms helpers)."""
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
                "eta": self._eta,
                "max_slew_per_second_ms": self._max_slew_per_second_ms,
                "coap_timeout_s": self._coap_timeout,
                "jitter_window": self._jitter_window,
                "min_samples_for_jitter": self._min_samples_for_jitter,
                "min_samples_before_log": self._min_samples_before_log,
                "bootstrap_theta_max_s": self._bootstrap_theta_max_s,
                "aggregate_updates": self._aggregate_updates,
            },
        }

    def inject_disturbance(self, delta_s: float) -> None:
        """Directly disturb the local offset by delta_s seconds (testing)."""
        self._offset += float(delta_s)
        print(f"[{self.node_id}] inject_disturbance: delta={delta_s*1000.0:.1f} ms, new_offset={self._offset*1000.0:.1f} ms")

    # -----------------------------------------------------------------
    # Robust statistics
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_iqr(samples: List[float]) -> float:
        """Compute IQR (Q3-Q1) with simple linear interpolation. samples are seconds."""
        n = len(samples)
        if n < 2:
            return 0.0
        xs = sorted(samples)

        def q(qv: float) -> float:
            pos = qv * (n - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return xs[lo]
            frac = pos - lo
            return xs[lo] * (1.0 - frac) + xs[hi] * frac

        return q(0.75) - q(0.25)

    def _update_link_jitter(self, peer_id: str, rtt_s: float) -> None:
        """Update rolling RTT buffer and sigma proxy (seconds)."""
        st = self._peer.setdefault(peer_id, PeerStats())
        st.rtt_samples.append(rtt_s)
        if len(st.rtt_samples) > self._jitter_window:
            st.rtt_samples.pop(0)

        if len(st.rtt_samples) >= self._min_samples_for_jitter:
            iqr = self._compute_iqr(st.rtt_samples)
            sigma = max(0.7413 * iqr, 1e-6)
            st.sigma = sigma

    def _weight_for_peer(self, peer_id: str) -> float:
        """Return inverse-variance-like weight from sigma (capped)."""
        st = self._peer.get(peer_id)
        if not st or st.sigma is None:
            return 1.0
        w = 1.0 / (st.sigma * st.sigma)
        return min(w, self._max_weight)

    # -----------------------------------------------------------------
    # Bootstrap & update logic
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
        One-time bootstrap:
            theta ≈ o_peer - o_self
            => o_self ≈ o_peer - theta   (rearrange)

        NOTE: In your previous version you used o_peer + theta, which is the opposite sign
        given theta is 'peer minus self'. This corrected sign matters a lot.
        """
        if not self._bootstrap_allowed(peer_id, theta_s):
            return False

        old = self._offset
        # theta = o_peer - o_self  =>  o_self = o_peer - theta
        self._offset = peer_offset_s - theta_s
        self._bootstrapped = True

        print(
            f"[{self.node_id}] BOOTSTRAP with {peer_id}: "
            f"theta={theta_s*1000.0:.3f} ms, old_offset={old*1000.0:.3f} ms, new_offset={self._offset*1000.0:.3f} ms"
        )
        return True

    def _compute_delta_from_measurement(self, peer_id: str, peer_offset_s: float, theta_s: float) -> Tuple[float, float, float]:
        """
        Compute correction delta (seconds) from a single peer measurement.

        We want: o_self ≈ o_peer - theta
        target = o_peer - theta
        error  = o_self - target
        delta  = -eta * w * error

        Returns: (error_s, raw_delta_s, delta_s_clipped)
        """
        target = peer_offset_s - theta_s
        error = self._offset - target

        w = self._weight_for_peer(peer_id)
        raw_delta = -self._eta * w * error

        # per-peer slew clipping based on dt since last update for that peer
        now_m = time.monotonic()
        st = self._peer.setdefault(peer_id, PeerStats())
        last = st.last_update_mono if st.last_update_mono is not None else now_m
        dt = max(now_m - last, 1e-3)
        st.last_update_mono = now_m

        if self._max_slew_per_second_ms > 0.0:
            max_step = (self._max_slew_per_second_ms / 1000.0) * dt
            delta = max(-max_step, min(max_step, raw_delta))
        else:
            delta = raw_delta

        return error, raw_delta, delta

    def _apply_delta(self, delta_s: float, dt_s: float) -> None:
        """Apply delta to offset with optional drift damping."""
        self._offset += delta_s

        if self._drift_damping > 0.0:
            # small linearized exponential decay towards 0
            self._offset *= max(0.0, 1.0 - self._drift_damping * dt_s)

    def _should_log(self, peer_id: str) -> bool:
        """DB/Web-UI gating: only log after enough good samples and after bootstrap."""
        if self._storage is None:
            return False
        if self._is_root:
            # root can log immediately if you want, but keep same gating
            pass
        st = self._peer.get(peer_id)
        if not st:
            return False
        return self._bootstrapped and (st.good_samples >= self._min_samples_before_log)

    def _log_sync_sample(
        self,
        peer: str,
        rtt_s: float,
        theta_s: float,
        error_s: Optional[float],
        delta_s: Optional[float],
        did_bootstrap: bool,
    ) -> None:
        """Write a sync sample into ntp_reference (if Storage exists and gating allows it)."""
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
        Send periodic beacons to all neighbors and update the local offset.

        Protocol (JSON)
        --------------
        Request payload:
            {
              "src": "<self id>",
              "dst": "<peer id>",
              "t1": <self_monotonic_seconds>,
              "boot_epoch": <self_boot_epoch_seconds>,   # optional but strongly recommended
              "offset": <self_offset_seconds>
            }

        Response payload:
            {
              "t2": <peer_monotonic_seconds>,
              "t3": <peer_monotonic_seconds>,
              "boot_epoch": <peer_boot_epoch_seconds>,   # optional but strongly recommended
              "offset": <peer_offset_seconds>
            }

        Time lifting
        ------------
        If both sides provide boot_epoch, we compute:
            t_epoch = t_mono + boot_epoch
        and run NTP math on epoch-like timestamps (common origin).
        Otherwise we fall back to raw monotonic (works only if all nodes booted "together").

        Update strategy
        ---------------
        - For each peer we compute theta and rtt.
        - We update jitter sigma from RTT.
        - We compute a candidate delta for that peer.
        - If aggregate_updates is enabled, we combine all peer deltas into one global delta
          (weighted average) and apply once per round with a global slew limit.
          This reduces oscillations when you have multiple peers.

        Logging
        -------
        Samples are written to the DB only after:
          - this node has bootstrapped, and
          - the peer has at least min_samples_before_log successful beacons.
        """
        if IS_WINDOWS or self._is_root:
            return

        deltas: List[Tuple[str, float, float, float, bool]] = []
        # tuple: (peer_id, rtt_s, theta_s, delta_s, did_bootstrap)

        for peer_id in self.neighbors:
            ip = self.neighbor_ips.get(peer_id)
            if not ip:
                continue

            uri = f"coap://{ip}/sync/beacon"

            # Local timestamps for NTP math
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

            # Decode response
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

            # Lift timestamps into a common epoch-like scale if possible
            if peer_boot_epoch is not None:
                t1 = t1_m + self._boot_epoch
                t4 = t4_m + self._boot_epoch
                t2 = t2_m + peer_boot_epoch
                t3 = t3_m + peer_boot_epoch
            else:
                # Fallback (not great across different boot times)
                t1, t4, t2, t3 = t1_m, t4_m, t2_m, t3_m

            # NTP formulas
            rtt = (t4 - t1) - (t3 - t2)
            theta = ((t2 - t1) + (t3 - t4)) / 2.0  # ≈ (peer_clock - self_clock)

            # Update per-peer stats
            st = self._peer.setdefault(peer_id, PeerStats())
            st.good_samples += 1
            st.theta_last = theta
            st.last_theta_epoch = time.time()
            self._update_link_jitter(peer_id, rtt)

            # Bootstrap if allowed
            did_bootstrap = self._try_bootstrap(peer_id, peer_offset, theta)

            # Compute delta (unless bootstrap happened)
            error = None
            raw_delta = None
            delta = None

            if not did_bootstrap:
                error, raw_delta, delta = self._compute_delta_from_measurement(peer_id, peer_offset, theta)
                deltas.append((peer_id, rtt, theta, delta, False))
            else:
                deltas.append((peer_id, rtt, theta, 0.0, True))

            # Short console summary
            sigma_ms = (st.sigma * 1000.0) if st.sigma is not None else None
            sigma_str = f"{sigma_ms:.3f} ms" if sigma_ms is not None else "n/a"
            print(
                f"[{self.node_id}] sync with {peer_id}: "
                f"theta={theta*1000.0:.3f} ms, rtt={rtt*1000.0:.3f} ms, sigma={sigma_str}"
            )

            # Per-peer logging (gated)
            self._log_sync_sample(
                peer=peer_id,
                rtt_s=rtt,
                theta_s=theta,
                error_s=error,
                delta_s=delta,
                did_bootstrap=did_bootstrap,
            )

        # Apply aggregated update once per round (recommended)
        if self._aggregate_updates:
            self._apply_aggregated_deltas(deltas)

    def _apply_aggregated_deltas(self, deltas: List[Tuple[str, float, float, float, bool]]) -> None:
        """
        Combine peer deltas into one global delta (weighted average) and apply once.
        This reduces multi-peer tug-of-war oscillations.

        deltas: list of (peer_id, rtt_s, theta_s, delta_s, did_bootstrap)
        """
        # Only consider non-bootstrap deltas
        candidates = [(pid, d) for (pid, _rtt, _th, d, did_bs) in deltas if (not did_bs)]
        if not candidates:
            return

        # Weighted average
        num = 0.0
        den = 0.0
        for pid, d in candidates:
            w = self._weight_for_peer(pid)
            num += w * d
            den += w

        if den <= 0.0:
            return

        delta = num / den

        # Global slew limit based on dt since last global update
        now_m = time.monotonic()
        last_m = self._last_global_update_mono if self._last_global_update_mono is not None else now_m
        dt = max(now_m - last_m, 1e-3)
        self._last_global_update_mono = now_m

        if self._max_slew_per_second_ms > 0.0:
            max_step = (self._max_slew_per_second_ms / 1000.0) * dt
            delta = max(-max_step, min(max_step, delta))

        self._apply_delta(delta, dt)

        print(
            f"[{self.node_id}] aggregate-update: "
            f"delta={delta*1000.0:.3f} ms, offset={self._offset*1000.0:.3f} ms (dt={dt:.3f}s)"
        )
