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
    Baryzentrischer Time-Sync für das Mesh.

    Idee:
      - Lokale Zeit: mono(t) = time.monotonic()
      - Mesh-Zeit:   mesh_time(t) = mono(t) + offset

    Jedes Node i hat einen Offset o_i (in Sekunden).
    Die NTP-4-Timestamp-Formel liefert eine Schätzung

        θ ≈ t_peer - t_self  ≈ o_peer - o_self

    Also: θ ist "peer minus self".

    Wir halten o_self so, dass

        o_self ≈ o_peer + θ

    (d.h. self zieht sich zum Peer hin).

    Fehler (aus Sicht von self):

        error = o_self - (o_peer + θ)

    Kosten:
        E = 0.5 * error^2

    Gradient-Update:
        o_self <- o_self - η * w * error

    Features:
      - Rolling RTT-Historie pro Peer
      - IQR-basierte Jitter-Schätzung σ_ij
      - Gewichtete Updates (1/σ²)
      - Slew-Limit (ms/s)
      - Optionaler Drift-Dämpfer
      - Bootstrap: einmaliger harter Sprung auf NTP-Schätzung
      - Optionales Logging pro Sync in SQLite (über Storage)
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
        self.neighbors = neighbors
        self.neighbor_ips = neighbor_ips
        self._storage = storage

        if sync_cfg is None:
            sync_cfg = {}

        # Root-Flag (z.B. für Node C)
        self._is_root = bool(sync_cfg.get("is_root", False))

        # ----------------------------------------------------------
        # Konfiguration aus JSON
        # ----------------------------------------------------------
        # Zufälliger Initial-Offset in ± initial_offset_ms (ms)
        initial_offset_ms = float(sync_cfg.get("initial_offset_ms", 200.0))

        # Rolling-Window-Größe für RTT / Jitter
        self._jitter_window = int(sync_cfg.get("jitter_window", 30))

        # Ab wie vielen Samples IQR/σ berechnet wird
        self._min_samples_for_jitter = int(
            sync_cfg.get("min_samples_for_jitter", 10)
        )

        # Lernrate η (0 < eta <= 1, aber realistisch << 1)
        self._eta = float(sync_cfg.get("eta", 0.02))

        # Slew-Limit in ms/s (Konfig-Einheit)
        self._max_slew_per_second_ms = float(
            sync_cfg.get("max_slew_per_second_ms", 10.0)
        )

        # CoAP-Timeout in Sekunden
        self._coap_timeout = float(sync_cfg.get("coap_timeout_s", 0.5))

        # Globaler Drift-Dämpfer (0 = aus, z.B. 0.01 = 1%/s)
        self._drift_damping = float(sync_cfg.get("drift_damping", 0.0))

        # Bootstrap-Threshold in Sekunden: |theta| <= threshold -> harter Bootstrap erlaubt
        self._bootstrap_theta_max_s = float(
            sync_cfg.get("bootstrap_theta_max_s", 0.0)  # 0 = immer erste Messung
        )

        # Zufälliger Initial-Offset (Sekunden) nur für Nicht-Root
        if self._is_root:
            # Root: Offset fest auf 0 (Meshzeit = Monotonic)
            self._offset = 0.0
            self._bootstrapped = True  # Root braucht keinen Bootstrap
        else:
            self._offset = random.uniform(
                -initial_offset_ms / 1000.0,
                +initial_offset_ms / 1000.0,
            )
            self._bootstrapped = False

        # letzte θ (NTP-Schätzung: o_peer - o_self) pro Peer
        self._peer_theta: Dict[str, float] = {}

        # RTT-Samples (Sekunden)
        self._peer_rtt_samples: Dict[str, List[float]] = {}

        # Jitter σ_ij (Sekunden)
        self._peer_sigma: Dict[str, float] = {}

        # Zeitstempel des letzten Updates pro Peer (für dt und Slew-Limit)
        self._last_update_ts: Dict[str, float] = {}

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------

    def mesh_time(self) -> float:
        """Aktuelle Mesh-Zeit in Sekunden (monotonic + Offset)."""
        return time.monotonic() + self._offset

    def get_offset(self) -> float:
        """Aktuellen Offset in Sekunden (für Beacon-Response)."""
        return self._offset

    def inject_disturbance(self, delta: float) -> None:
        """
        Chaos-Button: Offset direkt stören (delta in Sekunden).
        Praktisch zum Testen der Konvergenz.
        """
        self._offset += delta
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
          - mesh_time
          - offset_estimate (Sekunden + ms)
          - monotonic_now
          - peer_theta / peer_theta_ms (NTP-θ = o_peer - o_self)
          - peer_sigma / peer_sigma_ms
          - neighbors
          - sync_config (wichtige Parameter)
        """
        monotonic_now = time.monotonic()

        peer_theta_s = dict(self._peer_theta)
        peer_theta_ms = {k: v * 1000.0 for k, v in peer_theta_s.items()}

        peer_sigma_s = dict(self._peer_sigma)
        peer_sigma_ms = {k: v * 1000.0 for k, v in peer_sigma_s.items()}

        return {
            "node_id": self.node_id,
            "mesh_time": self.mesh_time(),
            "offset_estimate": self._offset,
            "offset_estimate_ms": self._offset * 1000.0,
            "monotonic_now": monotonic_now,
            "peer_theta": peer_theta_s,
            "peer_theta_ms": peer_theta_ms,
            "peer_sigma": peer_sigma_s,
            "peer_sigma_ms": peer_sigma_ms,
            "neighbors": list(self.neighbors),
            "sync_config": {
                "jitter_window": self._jitter_window,
                "min_samples_for_jitter": self._min_samples_for_jitter,
                "eta": self._eta,
                "max_slew_per_second_ms": self._max_slew_per_second_ms,
                "coap_timeout_s": self._coap_timeout,
                "drift_damping": self._drift_damping,
                "bootstrap_theta_max_s": self._bootstrap_theta_max_s,
            },
        }

    # --------------------------------------------------------------
    # Statistik-Helfer
    # --------------------------------------------------------------

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
            sigma = 0.7413 * iqr  # IQR -> σ Approx.
            self._peer_sigma[peer] = max(sigma, 1e-6)

    # --------------------------------------------------------------
    # Bootstrap (NTP-θ)
    # --------------------------------------------------------------

    def _try_bootstrap(self, peer: str, peer_offset: float, theta: float) -> bool:
        """
        Einmaliger harter Bootstrap für dieses Node (self).

        θ ≈ o_peer - o_self

        Wir wollen in den Konsens:

            o_self ≈ o_peer + θ   (self zieht zu peer hin)

        Also setzen wir beim Bootstrap:

            o_self <- o_peer + θ
        """
        # Root lässt seinen Offset nie von Peers verändern
        if self._is_root:
            return False

        if self._bootstrapped:
            return False

        # Optionaler Threshold: wenn konfiguriert und θ zu groß, kein Bootstrap
        if self._bootstrap_theta_max_s > 0.0:
            if abs(theta) > self._bootstrap_theta_max_s:
                return False

        old_offset = self._offset
        new_offset = peer_offset + theta
        self._offset = new_offset
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

    # --------------------------------------------------------------
    # Symmetrisches Update (Herzstück)
    # --------------------------------------------------------------

    def _symmetrical_update(self, peer: str, peer_offset: float, theta: float):
        """
        Symmetrisches pairwise Update aus Sicht dieses Nodes (self).

        θ ≈ o_peer - o_self

        Ziel:
            o_self ≈ o_peer + θ

        Fehler:
            error = o_self - (o_peer + θ)

        Gradient-Schritt:
            o_self <- o_self - η * w * error

        Alle Größen (Offsets, θ, error, delta) sind in Sekunden.

        Gibt (error, raw_delta, delta) in Sekunden zurück.
        """

        if self._is_root:
            # Root ändert seinen Offset nicht durch Peers
            return 0.0, 0.0, 0.0

        # Fehler (Sekunden)
        target = peer_offset + theta
        error = self._offset - target

        # Gewichtung nach Link-Qualität (σ in Sekunden)
        sigma = self._peer_sigma.get(peer)
        if sigma is not None and sigma > 1e-6:
            weight = 1.0 / (sigma * sigma)
        else:
            weight = 1.0

        # Theoretischer Schritt (Sekunden)
        raw_delta = -self._eta * weight * error

        # dt pro Peer
        now = time.monotonic()
        last = self._last_update_ts.get(peer, now)
        dt = now - last
        if dt <= 0.0:
            dt = 1e-3
        self._last_update_ts[peer] = now

        # Slew-Limit: max_slew_per_second_ms -> Sekunden pro Sekunde
        if self._max_slew_per_second_ms > 0.0:
            max_step_s = (self._max_slew_per_second_ms / 1000.0) * dt
            if raw_delta > max_step_s:
                delta = max_step_s
            elif raw_delta < -max_step_s:
                delta = -max_step_s
            else:
                delta = raw_delta
        else:
            delta = raw_delta

        # Offset aktualisieren
        self._offset += delta

        # Drift-Dämpfung ~ exp(-λ dt) (hier linearisiert)
        if self._drift_damping > 0.0:
            factor = max(0.0, 1.0 - self._drift_damping * dt)
            self._offset *= factor

        # Debug
        print(
            "[{}] sym-update with {}: error={:.3f} ms, raw_delta={:.3f} ms, "
            "delta={:.3f} ms, offset={:.3f} ms, target={:.3f} ms".format(
                self.node_id,
                peer,
                error * 1000.0,
                raw_delta * 1000.0,
                delta * 1000.0,
                self._offset * 1000.0,
                target * 1000.0,
            )
        )

        return error, raw_delta, delta

    # --------------------------------------------------------------
    # Logging-Helfer für Web-UI
    # --------------------------------------------------------------

    def _log_sync_sample(
        self,
        peer: str,
        rtt: float,
        theta: float,
        error: Optional[float],
        delta: Optional[float],
        did_bootstrap: bool,
    ) -> None:
        """
        Schreibt einen Sync-Sample in die ntp_reference-Tabelle, falls Storage vorhanden.

        Alle Zeiten in Sekunden.
        """
        # DB-Gating: erst loggen, wenn genügend RTT-Samples für diesen Peer da sind
        samples = self._peer_rtt_samples.get(peer, [])
        if len(samples) < self._min_samples_for_jitter:
            return

        if self._storage is None:
            return

        t_wall = time.time()
        t_mono = time.monotonic()
        t_mesh = self.mesh_time()
        offset = self._offset
        err_mesh_vs_wall = t_mesh - t_wall
        sigma = self._peer_sigma.get(peer)

        try:
            self._storage.insert_ntp_reference(
                node_id=self.node_id,
                t_wall=t_wall,
                t_mono=t_mono,
                t_mesh=t_mesh,
                offset=offset,
                err_mesh_vs_wall=err_mesh_vs_wall,
                peer_id=peer,
                theta_ms=(theta * 1000.0) if theta is not None else None,
                rtt_ms=(rtt * 1000.0) if rtt is not None else None,
                sigma_ms=(sigma * 1000.0) if sigma is not None else None,
            )
        except Exception as e:
            print(f"[{self.node_id}] _log_sync_sample failed: {e}")

    # --------------------------------------------------------------
    # Beacon-Versand
    # --------------------------------------------------------------

    async def send_beacons(self, client_ctx: aiocoap.Context) -> None:
        """
        Periodische Beacons zu allen Nachbarn senden.

        Für jeden Nachbarn j:
          - t1 = monotonic()      (bei self)
          - POST /sync/beacon mit {src, dst, t1, offset_self}
          - Antwort: {t2, t3, offset_peer}
          - t4 = monotonic()      (bei self)
          - NTP-Formeln:

                θ = ((t2 - t1) + (t3 - t4)) / 2  ≈ o_peer - o_self
                rtt = (t4 - t1) - (t3 - t2)

          - Jitter-Update, Bootstrap, symmetrisches Update, Logging.
        """
        if IS_WINDOWS:
            return

        if self._is_root:
            return  # Root ist nur Server, kein Client für Sync

        for peer in self.neighbors:
            ip = self.neighbor_ips.get(peer)
            if not ip:
                continue

            uri = f"coap://{ip}/sync/beacon"

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
            theta = ((t2 - t1) + (t3 - t4)) / 2.0  # ≈ o_peer - o_self

            # Statistik-Update
            self._update_rtt_sigma(peer, rtt)
            self._peer_theta[peer] = theta

            # Bootstrap zuerst (einmaliger harter Sprung)
            did_bootstrap = self._try_bootstrap(peer, peer_offset, theta)

            error = None
            delta = None

            # Danach feines Nachregeln
            if not did_bootstrap:
                error, raw_delta, delta = self._symmetrical_update(
                    peer, peer_offset, theta
                )
            else:
                raw_delta = None

            # Kurzsummary
            if peer in self._peer_sigma:
                sigma_ms = self._peer_sigma[peer] * 1000.0
                sigma_str = f"{sigma_ms:.3f} ms"
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

            # Sample in DB loggen (für Web-UI & Auswertung)
            self._log_sync_sample(
                peer=peer,
                rtt=rtt,
                theta=theta,
                error=error,
                delta=delta,
                did_bootstrap=did_bootstrap,
            )

    def is_warmed_up(self, min_samples: Optional[int] = None) -> bool:
        # Root ist per Definition stabil
        if self._is_root:
            return True

        # Erst nach Bootstrap macht Telemetrie Sinn
        if not self._bootstrapped:
            return False

        ms = int(min_samples or self._min_samples_for_jitter)

        # Wenn keine Neighbors, dann kann man nicht “warm” werden
        if not self.neighbors:
            return False

        # Warm, wenn wir pro Neighbor genug RTT-Samples haben
        for p in self.neighbors:
            if len(self._peer_rtt_samples.get(p, [])) < ms:
                return False
        return True
