# mesh/sync.py

import time
import asyncio
import random
import platform
import math
from typing import List, Dict, Optional

import aiocoap
import json

IS_WINDOWS = platform.system() == "Windows"


class SyncModule:
    """
    Time sync module.

    - mesh_time() = monotonic + offset
    - offset wird aus 4-Timestamp-Messungen zu den Nachbarn geschätzt
    - Unter Windows ist send_beacons() ein No-Op (Dev-Mode ohne echte Peers)

    Features:
      - per-peer RTT history
      - IQR-basierte Jitter-Schätzung σ_ij
      - inverse-variance Fusion über alle Peers
      - slew-limited Offset-Updates

    Parameter kommen (optional) aus einer globalen "sync"-Config.
    """

    def __init__(
        self,
        node_id: str,
        neighbors: List[str],
        neighbor_ips: Dict[str, str],
        sync_cfg: Optional[Dict[str, float]] = None,
    ):
        self.node_id = node_id
        self.neighbors = neighbors
        self.neighbor_ips = neighbor_ips  # e.g. {"B": "192.168.1.21", ...}

        # --------------------------------------------------------------
        # Parameter aus globaler Config (falls vorhanden)
        # --------------------------------------------------------------
        if sync_cfg is None:
            sync_cfg = {}

        # Range für zufälligen Start-Offset (± initial_offset_ms)
        initial_offset_ms = float(sync_cfg.get("initial_offset_ms", 200.0))

        # Rolling-Window-Grösse für RTT / Jitter
        self._jitter_window = int(sync_cfg.get("jitter_window", 30))

        # Ab wie vielen Samples IQR/σ berechnet wird
        self._min_samples_for_jitter = int(sync_cfg.get("min_samples_for_jitter", 10))

        # Slew-Limit in ms/s → Sekunden/s
        max_slew_ms = float(sync_cfg.get("max_slew_per_second_ms", 5.0))
        self._max_slew_per_second = max_slew_ms / 1000.0  # s Offset pro s

        # CoAP-Timeout (Sekunden) für Beacon-Requests
        self._coap_timeout = float(sync_cfg.get("coap_timeout_s", 0.5))

        # Zufälliger Initial-Offset, damit Konvergenz sichtbar ist
        self._offset = random.uniform(
            -initial_offset_ms / 1000.0,
            +initial_offset_ms / 1000.0,
        )

        # Letzte Offset-Schätzung pro Nachbar (θ_ij)
        self._peer_offsets: Dict[str, float] = {}

        # Rolling RTT-Samples pro Nachbar (δ_ij)
        self._peer_rtt_samples: Dict[str, list[float]] = {}

        # Geschätzter Jitter σ_ij (Sekunden) pro Nachbar
        self._peer_sigma: Dict[str, float] = {}

        # Für zeitbasierte Slew-Limitierung
        self._last_update_time = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mesh_time(self) -> float:
        """Return the node's current mesh time."""
        return time.monotonic() + self._offset

    def inject_disturbance(self, delta: float):
        """Chaos-Button: Offset direkt stören (z.B. über /sync/disturb)."""
        self._offset += delta
        self._last_update_time = time.monotonic()
        print(
            "[{}] inject_disturbance: delta={:.1f} ms, new offset={:.1f} ms".format(
                self.node_id, delta * 1000.0, self._offset * 1000.0
            )
        )

    def status_snapshot(self) -> dict:
        """Für /status Endpoint."""
        return {
            "node_id": self.node_id,
            "offset_estimate": self._offset,
            "peer_offsets": self._peer_offsets,
            "peer_sigma": self._peer_sigma,
            "neighbors": self.neighbors,
        }

    # ------------------------------------------------------------------
    # Interne Helfer: Statistik & Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iqr(samples: list) -> float:
        """Compute IQR (Q3 - Q1) mit einfacher linearer Interpolation."""
        n = len(samples)
        if n < 2:
            return 0.0

        xs = sorted(samples)

        def quantile(q: float) -> float:
            # q in [0,1]
            if n == 1:
                return xs[0]
            pos = q * (n - 1)
            lo = int(math.floor(pos))
            hi = int(math.ceil(pos))
            if lo == hi:
                return xs[lo]
            frac = pos - lo
            return xs[lo] * (1.0 - frac) + xs[hi] * frac

        q1 = quantile(0.25)
        q3 = quantile(0.75)
        return q3 - q1

    def _update_peer_stats(self, peer: str, theta: float, rtt: float):
        """
        Update per-peer θ_ij, RTT-Historie, Jitter σ_ij
        und danach globale Offset-Fusion mit Slew-Limit.
        """
        # Letzte Offset-Schätzung für diesen Peer
        self._peer_offsets[peer] = theta

        # RTT-Historie aktualisieren
        buf = self._peer_rtt_samples.setdefault(peer, [])
        buf.append(rtt)
        if len(buf) > self._jitter_window:
            buf.pop(0)  # nur die letzten N Werte behalten

        # Wenn genug Samples vorhanden sind → σ_ij über IQR schätzen
        if len(buf) >= self._min_samples_for_jitter:
            iqr = self._compute_iqr(buf)
            # Robuste σ-Schätzung aus IQR (für ungefähr normalverteilten Lärm)
            sigma = 0.7413 * iqr
            sigma = max(sigma, 1e-6)  # Degenerates vermeiden
            self._peer_sigma[peer] = sigma
        # sonst: alte σ beibehalten (falls vorhanden), Peer wird wie "gleich" gewichtet

        # Danach globalen Offset neu berechnen
        self._recompute_fused_offset()

    def _recompute_fused_offset(self):
        """
        Fused Offset θ̂(i) via inverse-variance Fusion:

            θ̂(i) = Σ_j θ_ij / σ_ij²  /  Σ_j 1 / σ_ij²

        Falls noch keine σ_ij vorhanden → einfacher Mittelwert der Peers.

        Dann Slew-Limit anwenden: max X ms/s anhand der vergangenen Zeit.
        """
        if not self._peer_offsets:
            return  # noch keine Messungen

        # 1) Inverse-Variance-Gewichtung für Peers mit gültigem σ_ij
        weighted_sum = 0.0
        weight_total = 0.0

        for peer, theta in self._peer_offsets.items():
            sigma = self._peer_sigma.get(peer)
            if sigma is not None and sigma > 1e-6 and math.isfinite(sigma):
                w = 1.0 / (sigma * sigma)
                weighted_sum += theta * w
                weight_total += w

        if weight_total > 0.0:
            target_offset = weighted_sum / weight_total
        else:
            # 2) Fallback: einfacher Mittelwert aller θ_ij
            target_offset = sum(self._peer_offsets.values()) / float(len(self._peer_offsets))

        # 3) Slew-Limit basierend auf vergangener Zeit
        now = time.monotonic()
        dt = now - self._last_update_time
        if dt <= 0.0:
            dt = 1e-3  # kleine positive Zeit, um Edge-Cases zu vermeiden

        max_delta = self._max_slew_per_second * dt  # maximaler Offset-Schritt (Sekunden)

        delta = target_offset - self._offset
        if abs(delta) <= max_delta:
            # wir können das Ziel direkt erreichen
            self._offset = target_offset
        else:
            # nur max_delta in Richtung Ziel bewegen
            self._offset += math.copysign(max_delta, delta)

        self._last_update_time = now

        # Debug-Ausgabe (bei Bedarf auskommentieren)
        print(
            "[{}] fused offset update: target={:.2f} ms, new_offset={:.2f} ms, dt={:.3f}s".format(
                self.node_id,
                target_offset * 1000.0,
                self._offset * 1000.0,
                dt,
            )
        )

    # ------------------------------------------------------------------
    # Beacon sending (public async API)
    # ------------------------------------------------------------------

    async def send_beacons(self, client_ctx: aiocoap.Context):
        """
        Wird periodisch aus MeshNode.sync_loop() aufgerufen.

        Für jeden Nachbarn:
          - POST /sync/beacon mit t1 und src id senden
          - Antwort mit t2, t3 empfangen
          - t4 lokal messen
          - θ_ij und δ_ij (RTT) berechnen
          - Rolling-Statistiken und fused Offset updaten

        Unter Windows ist das ein No-Op (Dev-Mode).
        """
        if IS_WINDOWS:
            return

        for peer in self.neighbors:
            ip = self.neighbor_ips.get(peer)
            if not ip:
                continue

            uri = "coap://{}/sync/beacon".format(ip)

            t1 = time.monotonic()
            payload = {
                "src": self.node_id,
                "dst": peer,
                "t1": t1,
            }

            req = aiocoap.Message(
                code=aiocoap.POST,
                uri=uri,
                payload=json.dumps(payload).encode("utf-8"),
            )

            try:
                # CoAP-Request mit Timeout
                req_ctx = client_ctx.request(req)
                resp = await asyncio.wait_for(
                    req_ctx.response,
                    timeout=self._coap_timeout,
                )
            except asyncio.TimeoutError:
                print("[{}] beacon to {} timed out after {:.3f}s".format(
                    self.node_id, peer, self._coap_timeout
                ))
                continue
            except Exception as e:
                print("[{}] beacon to {} failed: {}".format(self.node_id, peer, e))
                continue

            t4 = time.monotonic()

            try:
                data = json.loads(resp.payload.decode("utf-8"))
                t2 = float(data["t2"])
                t3 = float(data["t3"])
            except Exception as e:
                print("[{}] invalid beacon reply from {}: {}".format(self.node_id, peer, e))
                continue

            # NTP-Style 4-Timestamp Formeln
            rtt = (t4 - t1) - (t3 - t2)            # δ_ij (round-trip minus serverzeit)
            theta = ((t2 - t1) + (t3 - t4)) / 2.0  # θ_ij (Offset-Schätzung)

            # Per-Peer-Statistiken + fused Offset updaten
            self._update_peer_stats(peer, theta, rtt)

            # Debug-Ausgabe für diesen Link
            sigma_str = "n/a"
            if peer in self._peer_sigma:
                sigma_str = "{:.2f} ms".format(self._peer_sigma[peer] * 1000.0)

            print(
                "[{}] sync with {}: theta={:.2f} ms, rtt={:.2f} ms, sigma={}, offset={:.2f} ms".format(
                    self.node_id,
                    peer,
                    theta * 1000.0,
                    rtt * 1000.0,
                    sigma_str,
                    self._offset * 1000.0,
                )
            )
