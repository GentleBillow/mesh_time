# -*- coding: utf-8 -*-
# mesh/kalman_controller.py

from __future__ import annotations
from typing import Optional
import numpy as np
from typing import Dict, Any, List, Tuple, Callable

Measurement = Tuple[str, float, float, float]
WeightFn = Callable[[str], float]
NoiseFn = Callable[[str, float], float]


def _median(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if (n % 2 == 1) else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def _p95(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    k = int(round(0.95 * (len(xs) - 1)))
    return xs[max(0, min(len(xs) - 1, k))]

class KalmanController:
    """
    Kalman Filter with state:
      x = [ offset, drift, bias_peer1, bias_peer2, ... ]

    Measurements:
      z_ij = peer_offset - theta_ij ≈ offset - bias_ij

    Gauges (optional, independent):
      - Offset gauge: anchors absolute offset to 0 (removes global offset ambiguity)
      - Bias gauge:   softly prefers per-peer bias ≈ 0 (prevents bias random-walk / overfitting)
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = dict(cfg)

        # Process noise
        self.q_offset = float(cfg.get("q_offset", 1e-9))
        self.q_drift  = float(cfg.get("q_drift", 1e-12))
        self.q_bias   = float(cfg.get("q_bias", 1e-14))

        # Measurement noise base
        self.r_base = float(cfg.get("r_base", 1e-6))

        # Gauge config
        self.use_offset_gauge = bool(cfg.get("use_offset_gauge", cfg.get("use_gauge", False)))
        self.r_offset_gauge   = float(cfg.get("r_offset_gauge", cfg.get("r_virtual", 1e-2)))

        self.use_bias_gauge   = bool(cfg.get("use_bias_gauge", False))
        self.r_bias_gauge     = float(cfg.get("r_bias_gauge", 1e-2))

        # Internal state (initialized lazily)
        self._x = None           # state vector
        self._P = None           # covariance
        self._peer_index = {}    # peer_id -> bias index

        self._last_diag = None

    # ------------------------------------------------------------

    def on_bootstrap(self) -> None:
        self._x = None
        self._P = None
        self._peer_index = {}
        self._last_diag = None

    def last_diag(self) -> Dict[str, Any]:
        return dict(self._last_diag or {})

    # ------------------------------------------------------------

    def _ensure_state(self, peers: List[str]) -> None:
        """Ensure state vector contains all peers in deterministic order."""
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

    @staticmethod
    def _kf_update(x: np.ndarray, P: np.ndarray, H: np.ndarray, z: float, R: float) -> tuple[np.ndarray, np.ndarray]:
        """Generic scalar measurement update."""
        y = np.array([[z]]) - (H @ x)
        S = (H @ P @ H.T) + np.array([[R]])
        K = P @ H.T @ np.linalg.inv(S)
        x = x + (K @ y)
        P = (np.eye(P.shape[0]) - (K @ H)) @ P
        return x, P

    # ------------------------------------------------------------

    def compute_delta(
        self,
        offset_s: float,
        measurements: List[Measurement],
        weight_fn: WeightFn,
        dt_s: float,
        noise_fn: Optional[NoiseFn] = None,
    ) -> float:

        if not measurements:
            return 0.0

        innov_ms_list: List[float] = []
        nis_list: List[float] = []
        r_list_ms2: List[float] = []
        n_meas = 0

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
        for _, bidx in self._peer_index.items():
            Q[2 + bidx, 2 + bidx] = self.q_bias * dt_s

        x = F @ x
        P = F @ P @ F.T + Q

        # ---------------- Update (per measurement) ----------------
        for peer_id, _rtt, theta, peer_offset in measurements:
            bidx = self._peer_index[peer_id]
            idx = 2 + bidx

            z = peer_offset - theta  # ≈ offset - bias_peer

            H = np.zeros((1, dim))
            H[0, 0] = 1.0
            H[0, idx] = -1.0

            if noise_fn is not None:
                R = float(noise_fn(peer_id, float(_rtt)))
            else:
                w = float(weight_fn(peer_id))
                R = float(self.r_base / max(w, 1e-9))

            # --- DIAG BEFORE UPDATE ---
            # innovation y = z - Hx
            y = float(z) - float((H @ x)[0, 0])
            S = float((H @ P @ H.T)[0, 0] + R)
            if S > 0:
                nis = (y * y) / S
                nis_list.append(float(nis))

            innov_ms_list.append(float(y * 1000.0))
            r_list_ms2.append(float(R * 1e6))  # sec^2 -> ms^2  ( (s*1000)^2 = s^2*1e6 )
            n_meas += 1

            x, P = self._kf_update(x, P, H, z=float(z), R=R)

        # ==========================================================
        # Optional gauges (independent)
        # ==========================================================

        # --- Offset gauge: offset ≈ 0
        if self.use_offset_gauge:
            H = np.zeros((1, dim))
            H[0, 0] = 1.0
            x, P = self._kf_update(x, P, H, z=0.0, R=float(self.r_offset_gauge))

        # --- Bias gauge: bias_peer ≈ 0  (applied per known peer)
        if self.use_bias_gauge and self._peer_index:
            for _peer, bidx in self._peer_index.items():
                idx = 2 + bidx
                H = np.zeros((1, dim))
                H[0, idx] = 1.0
                x, P = self._kf_update(x, P, H, z=0.0, R=float(self.r_bias_gauge))

        self._last_diag = {
            "n_meas": int(n_meas),
            "innov_med_ms": _median(innov_ms_list),
            "innov_p95_ms": _p95(innov_ms_list),
            "nis_med": _median(nis_list),
            "nis_p95": _p95(nis_list),
            "r_eff_ms2": _median(r_list_ms2),

            "x_offset_ms": float(x[0, 0] * 1000.0),
            "x_drift_ppm": float(x[1, 0] * 1e6),

            "p_offset_ms2": float(P[0, 0] * 1e6),  # s^2 -> ms^2
            "p_drift_ppm2": float(P[1, 1] * 1e12),  # (1e6)^2
        }

        # ---------------- Output ----------------
        old_offset = float(offset_s)
        new_offset = float(x[0, 0])

        self._x = x
        self._P = P

        return new_offset - old_offset

    # ------------------------------------------------------------

    def commit_applied_offset(self, offset_s: float) -> None:
        """
        HARD state alignment.
        Must be called after slew-limited offset was applied.
        """
        if self._x is not None:
            self._x[0, 0] = float(offset_s)

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
            "gauges": {
                "use_offset_gauge": self.use_offset_gauge,
                "r_offset_gauge": float(self.r_offset_gauge),
                "use_bias_gauge": self.use_bias_gauge,
                "r_bias_gauge": float(self.r_bias_gauge),
            }
        }
