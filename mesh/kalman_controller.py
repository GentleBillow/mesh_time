# -*- coding: utf-8 -*-
# mesh/kalman_controller.py

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple, Callable

Measurement = Tuple[str, float, float, float]
WeightFn = Callable[[str], float]


class KalmanController:
    """
    Kalman Filter with state:
      x = [ offset, drift, bias_peer1, bias_peer2, ... ]

    Measurements:
      z_ij = peer_offset - theta_ij ≈ offset - bias_ij
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg)

        # Process noise
        self.q_offset = float(cfg.get("q_offset", 1e-9))
        self.q_drift  = float(cfg.get("q_drift", 1e-12))
        self.q_bias   = float(cfg.get("q_bias", 1e-14))

        # Measurement noise base
        self.r_base = float(cfg.get("r_base", 1e-6))

        # Internal state (initialized lazily)
        self._x = None           # state vector
        self._P = None           # covariance
        self._peer_index = {}    # peer_id -> bias index

    # ------------------------------------------------------------

    def on_bootstrap(self) -> None:
        # Full reset on bootstrap
        self._x = None
        self._P = None
        self._peer_index = {}

    # ------------------------------------------------------------

    def _ensure_state(self, peers: List[str]) -> None:
        """
        Ensure state vector contains all peers in deterministic order.
        """
        for p in sorted(peers):
            if p not in self._peer_index:
                self._peer_index[p] = len(self._peer_index)

        n_bias = len(self._peer_index)
        dim = 2 + n_bias

        if self._x is None:
            self._x = np.zeros((dim, 1))
            self._P = np.eye(dim) * 1e-3
            return

        if self._x.shape[0] == dim:
            return

        # Expand state
        old_dim = self._x.shape[0]
        x_new = np.zeros((dim, 1))
        P_new = np.zeros((dim, dim))

        x_new[:old_dim, 0] = self._x[:, 0]
        P_new[:old_dim, :old_dim] = self._P

        for i in range(old_dim, dim):
            P_new[i, i] = 1e-3

        self._x = x_new
        self._P = P_new

    # ------------------------------------------------------------

    def compute_delta(
        self,
        offset_s: float,
        measurements: List[Measurement],
        weight_fn: WeightFn,
        dt_s: float,
    ) -> float:

        if not measurements:
            return 0.0

        peers = [m[0] for m in measurements]
        self._ensure_state(peers)

        x = self._x
        P = self._P

        dim = x.shape[0]

        # ---------------- Prediction ----------------
        F = np.eye(dim)
        F[0, 1] = dt_s  # offset += drift * dt

        Q = np.zeros((dim, dim))
        Q[0, 0] = self.q_offset * dt_s
        Q[1, 1] = self.q_drift * dt_s
        for _, idx in self._peer_index.items():
            Q[2 + idx, 2 + idx] = self.q_bias * dt_s

        x = F @ x
        P = F @ P @ F.T + Q

        # ---------------- Update (per measurement) ----------------
        for peer_id, _rtt, theta, peer_offset in measurements:
            bias_idx = self._peer_index[peer_id]
            idx = 2 + bias_idx

            z = peer_offset - theta

            H = np.zeros((1, dim))
            H[0, 0] = 1.0
            H[0, idx] = -1.0

            w = float(weight_fn(peer_id))
            R = np.array([[self.r_base / max(w, 1e-9)]])

            y = np.array([[z]]) - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)

            x = x + K @ y
            P = (np.eye(dim) - K @ H) @ P

        # ---------------- Output ----------------
        old_offset = offset_s
        new_offset = float(x[0, 0])

        self._x = x
        self._P = P

        return new_offset - old_offset

    # ------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        if self._x is None:
            return {"type": "kalman", "initialized": False}

        bias_ms = {
            peer: float(self._x[2 + idx, 0] * 1000.0)
            for peer, idx in self._peer_index.items()
        }

        return {
            "type": "kalman",
            "initialized": True,
            "offset_ms": float(self._x[0, 0] * 1000.0),
            "drift_ppm": float(self._x[1, 0] * 1e6),
            "bias_ms": bias_ms,
        }
