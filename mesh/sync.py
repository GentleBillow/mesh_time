# -*- coding: utf-8 -*-
# mesh/sync.py

import time
import asyncio
import random
import platform
import math
import json
from typing import List, Dict, Optional

import aiocoap

IS_WINDOWS = platform.system() == "Windows"


class SyncModule:
    """
    Baryzentrische Time-Sync für das Mesh.

    Grundidee:
      - Jede Node hat eine lokale Zeit: mono(t) = time.monotonic()
      - Mesh-Zeit:  mesh_time(t) = mono(t) + offset
      - Die Offsets werden durch paarweise Messungen θ_ij iterativ angepasst.

    Ziel:
      Minimiere für alle Kanten (i, j):

          Σ w_ij * ( (o_j - o_i) - θ_ij )²

      d.h. die Differenz der Offsets soll zu den gemessenen
      relativen Offsets θ_ij passen (NTP 4-Timestamp-Schätzung).

    Features:
      - Rolling RTT-Historie pro Peer
      - IQR-basierte Jitter-Schätzung σ_ij
      - Gewichtete Updates (1/σ²)
      - Slew-Limit in Sekunden pro Sekunde
      - Optionaler Drift-Dämpfer: Offsets werden sanft gegen 0 gezogen
      - Debug-freundlicher Status-Snapshot
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
        self.neighbor_ips = neighbor_ips

        # --------------------------------------------------------------
        # Konfiguration
        # --------------------------------------------------------------
        if sync_cfg is None:
            sync_cfg = {}

        # Bootstrap-Flag: einmaliger harter Sprung in die "richtige Region"
        self._bootstrapped = False

        # Optionaler Bootstrap-Threshold, um völlig kaputte Messungen zu ignorieren
        self._bootstrap_theta_max = float(sync_cfg.get("bootstrap_theta_max_s", 1.0))
        # 1.0 s = wir erlauben einen harten Sprung, solange |theta| < 1s

        # Zufälliger Start-Offset in ± initial_offset_ms
        initial_offset_ms = float(sync_cfg.get("initial_offset_ms", 200.0))

        # Rolling-Window-Größe für RTT / Jitter
        self._jitter_window = int(sync_cfg.get("jitter_window", 30))

        # Ab wie vielen Samples IQR/σ berechnet wird
        self._min_samples_for_jitter = int(
            sync_cfg.get("min_samples_for_jitter", 10)
        )

        # Lernrate η für symmetrisches Update (0 < eta <= 1)
        self._eta = float(sync_cfg.get("eta", 0.5))

        # Slew-Limit in ms/s -> Sekunden/Sekunde
        max_slew_ms = float(sync_cfg.get("max_slew_per_second_ms", 5.0))
        self._max_slew_per_second = max_slew_ms / 1000.0

        # CoAP-Timeout (Sekunden) für Beacon-Requests
        self._coap_timeout = float(sync_cfg.get("coap_timeout_s", 0.5))

        # Globaler Drift-Dämpfer (0 = aus, 0.001 = 0.1% pro Update)
        self._drift_damping = float(sync_cfg.get("drift_damping", 0.001))

        # Zufälliger Initial-Offset (in Sekunden)
        self._offset = random.uniform(
            -initial_offset_ms / 1000.0,
            +initial_offset_ms / 1000.0,
        )

        # Letzte NTP-Offset-Schätzungen θ_ij pro Nachbar
        self._peer_offsets: Dict[str, float] = {}

        # Rolling RTT-Samples pro Nachbar
        self._peer_rtt_samples: Dict[str, List[float]] = {}

        # Geschätzter Jitter σ_ij (Sekunden) pro Nachbar
        self._peer_sigma: Dict[str, float] = {}

        # Für zeitbasiertes Slew-Limit
        self._last_update_time = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mesh_time(self) -> float:
        """Aktuelle Mesh-Zeit in Sekunden (monotonic + Offset)."""
        return time.monotonic() + self._offset

    def get_offset(self) -> float:
        """Aktuellen Offset in Sekunden zurückgeben (für Beacon-Response)."""
        return self._offset

    def inject_disturbance(self, delta: float) -> None:
        """
        Chaos-Button: Offset direkt stören (delta in Sekunden).
        Praktisch zum Testen der Konvergenz.
        """
        self._offset += delta
        self._last_update_time = time.monotonic()
        print(
            "[{}] inject_disturbance: delta={:.1f} ms, new offset={:.1f} ms".format(
                self.node_id, delta * 1000.0, self._offset * 1000.0
            )
        )

    def status_snapshot(self) -> dict:
        """
        Status-Snapshot für /status Endpoint.

        Liefert:
          - node_id
          - offset_estimate (Sekunden + ms)
          - mesh_time
          - monotonic_now
          - peer_offsets (θ_ij in s + ms)
          - peer_sigma (σ_ij in s + ms)
          - neighbors
          - sync_config (relevante Parameter)
        """
        monotonic_now = time.monotonic()

        peer_offsets_s = dict(self._peer_offsets)
        peer_offsets_ms = {
            k: v * 1000.0 for (k, v) in self._peer_offsets.items()
        }

        peer_sigma_s = dict(self._peer_sigma)
        peer_sigma_ms = {
            k: v * 1000.0 for (k, v) in self._peer_sigma.items()
        }

        return {
            "node_id": self.node_id,
            "mesh_time": self.mesh_time(),
            "offset_estimate": self._offset,
            "offset_estimate_ms": self._offset * 1000.0,
            "monotonic_now": monotonic_now,
            "peer_offsets": peer_offsets_s,
            "peer_offsets_ms": peer_offsets_ms,
            "peer_sigma": peer_sigma_s,
            "peer_sigma_ms": peer_sigma_ms,
            "neighbors": list(self.neighbors),
            "sync_config": {
                "jitter_window": self._jitter_window,
                "min_samples_for_jitter": self._min_samples_for_jitter,
                "eta": self._eta,
                "max_slew_per_second": self._max_slew_per_second,
                "coap_timeout_s": self._coap_timeout,
                "drift_damping": self._drift_damping,
            },
        }

    # ------------------------------------------------------------------
    # Statistik-Helfer
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_iqr(samples: List[float]) -> float:
        """
        Compute IQR (Q3 - Q1) mit einfacher linearer Interpolation.
        """
        n = len(samples)
        if n < 2:
            return 0.0

        xs = sorted(samples)

        def quantile(q: float) -> float:
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

    def _update_rtt_sigma(self, peer: str, rtt: float) -> None:
        """
        Update RTT-Historie und Jitter σ_ij pro Peer.

        rtt: Round-Trip-Time in Sekunden.
        """
        buf = self._peer_rtt_samples.setdefault(peer, [])
        buf.append(rtt)
        if len(buf) > self._jitter_window:
            buf.pop(0)

        if len(buf) >= self._min_samples_for_jitter:
            iqr = self._compute_iqr(buf)
            sigma = 0.7413 * iqr  # IQR -> σ Approximation
            self._peer_sigma[peer] = max(sigma, 1e-6)

    def _try_bootstrap(self, peer: str, peer_offset: float, theta: float) -> bool:
        """
        Führt einmalig einen harten Bootstrap-Schritt aus, wenn:
          - noch nicht gebootstrapped
          - |theta| < bootstrap_theta_max_s (Messung scheint plausibel)

        Setzt o_i so, dass (peer_offset - o_i) + theta ≈ 0 gilt:
            o_i = peer_offset + theta

        Gibt True zurück, wenn Bootstrap gemacht wurde.
        """
        if self._bootstrapped:
            return False

        if abs(theta) > self._bootstrap_theta_max:
            # Messung zu wild, kein Bootstrap
            return False

        new_offset = peer_offset + theta
        old_offset = self._offset
        self._offset = new_offset
        self._last_update_time = time.monotonic()
        self._bootstrapped = True

        print(
            "[{}] BOOTSTRAP with {}: theta={:.3f} ms, old_offset={:.3f} ms, new_offset={:.3f} ms".format(
                self.node_id,
                peer,
                theta * 1000.0,
                old_offset * 1000.0,
                new_offset * 1000.0,
            )
        )
        return True


    # ------------------------------------------------------------------
    # Symmetrisches Update (Herzstück)
    # ------------------------------------------------------------------

    def _symmetrical_update(self, peer: str, peer_offset: float, theta: float) -> None:
        """
        Symmetrisches pairwise Update zur Minimierung von

            E_ij = ((o_i - o_j) - θ_ij)^2

        Zielbedingung:
            o_i - o_j ≈ θ_ij

        Wir machen einen Gradienten-Schritt auf o_i:

            error = (o_i - o_j) - θ_ij
            o_i   <- o_i - η * w * error

        mit optionalem Slew-Limit und Drift-Dämpfer.
        """
        # Fehler relativ zur Zielbedingung
        error = (self._offset - peer_offset) - theta

        # Gewichtung nach Link-Qualität (falls σ bekannt)
        sigma = self._peer_sigma.get(peer)
        if sigma is not None and sigma > 1e-6:
            weight = 1.0 / (sigma * sigma)
        else:
            weight = 1.0

        # "theoretischer" Gradient-Schritt (ohne Slew-Limit)
        raw_delta = -self._eta * weight * error

        # Slew-Limit relativ zu echter Zeit
        now = time.monotonic()
        dt = now - self._last_update_time
        if dt <= 0.0:
            dt = 1e-3

        max_delta = self._max_slew_per_second * dt
        if raw_delta > max_delta:
            delta = max_delta
        elif raw_delta < -max_delta:
            delta = -max_delta
        else:
            delta = raw_delta

        # Offset anpassen
        self._offset += delta

        # Optional: globaler Drift-Dämpfer (zieht offset langsam Richtung 0)
        if self._drift_damping > 0.0:
            self._offset *= (1.0 - self._drift_damping)

        # Zeitstempel updaten
        self._last_update_time = now

        # Debug-Output (einmal)
        print(
            "[{}] sym-update with {}: error={:.3f} ms, raw_delta={:.3f} ms, "
            "delta={:.3f} ms, offset={:.3f} ms".format(
                self.node_id,
                peer,
                error * 1000.0,
                raw_delta * 1000.0,
                delta * 1000.0,
                self._offset * 1000.0,
            )
        )

    # ------------------------------------------------------------------
    # Beacon-Versand
    # ------------------------------------------------------------------

    async def send_beacons(self, client_ctx: aiocoap.Context) -> None:
        """
        Periodische Beacons zu allen Nachbarn senden.

        Für jeden Nachbarn j:
          - t1 = monotonic()
          - POST /sync/beacon mit {src, dst, t1, offset_i}
          - Antwort: {t2, t3, offset_j}
          - t4 = monotonic()
          - NTP-Formeln -> θ_ij, RTT
          - Jitter-Update, symmetrisches Offset-Update

        Unter Windows: Dev-Mode -> no-op.
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
                "offset": self._offset,
            }

            req = aiocoap.Message(
                code=aiocoap.POST,
                uri=uri,
                payload=json.dumps(payload).encode("utf-8"),
            )

            try:
                req_ctx = client_ctx.request(req)
                resp = await asyncio.wait_for(
                    req_ctx.response,
                    timeout=self._coap_timeout,
                )
            except asyncio.TimeoutError:
                print(
                    "[{}] beacon to {} timed out after {:.3f}s".format(
                        self.node_id, peer, self._coap_timeout
                    )
                )
                continue
            except Exception as e:
                print("[{}] beacon to {} failed: {}".format(self.node_id, peer, e))
                continue

            t4 = time.monotonic()

            # Antwort dekodieren
            try:
                data = json.loads(resp.payload.decode("utf-8"))
                t2 = float(data["t2"])
                t3 = float(data["t3"])
                peer_offset = float(data["offset"])
            except Exception as e:
                print(
                    "[{}] invalid beacon reply from {}: {}".format(
                        self.node_id, peer, e
                    )
                )
                continue

            # NTP-Formeln
            rtt = (t4 - t1) - (t3 - t2)
            theta = ((t2 - t1) + (t3 - t4)) / 2.0

            # Statistik-Update
            self._update_rtt_sigma(peer, rtt)
            self._peer_offsets[peer] = theta

            # ZUERST: Bootstrap versuchen (einmaliger harter Sprung in die richtige Region)
            did_bootstrap = self._try_bootstrap(peer, peer_offset, theta)

            # Danach: reguläres symmetrisches Update (feines Nachregeln)
            if not did_bootstrap:
                self._symmetrical_update(peer, peer_offset, theta)

            # Debug-Kurzsummary
            if peer in self._peer_sigma:
                sigma_ms = self._peer_sigma[peer] * 1000.0
                sigma_str = "{:.3f} ms".format(sigma_ms)
            else:
                sigma_str = "n/a"

            print(
                "[{}] sync with {}: theta={:.3f} ms, rtt={:.3f} ms, sigma={}".format(
                    self.node_id,
                    peer,
                    theta * 1000.0,
                    rtt * 1000.0,
                    sigma_str,
                )
            )
