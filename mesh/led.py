# mesh/led.py
# -*- coding: utf-8 -*-

import time
import platform

try:
    import grovepi
    HAS_GROVEPI = True
except ImportError:
    HAS_GROVEPI = False

IS_WINDOWS = platform.system() == "Windows"


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class _BlinkCore:
    """
    Gemeinsame Logik: glättet t_mesh -> t_disp (Anzeigezeit),
    damit kleine Sync-Korrekturen nicht als LED-Phasen-Zittern sichtbar werden.
    """
    def __init__(self):
        self._t_disp = None          # geglättete Anzeigezeit (Sekunden)
        self._last_mono = None       # letzter monotonic Zeitpunkt (Sekunden)
        self._last_on = False

    def compute_on(
        self,
        t_mesh: float,
        *,
        period: float = 1.0,
        pulse_s: float = 0.05,              # 50ms Pulsfenster (optisch stabiler als 20ms)
        max_slew_ms_per_s: float = 250.0,   # wie du’s schon nutzt: 250ms pro Sekunde
        snap_if_far_s: float = 2.0,         # wenn wir zu weit weg sind -> sofort "snappen"
    ) -> bool:
        now_mono = time.monotonic()

        if self._t_disp is None:
            self._t_disp = t_mesh
            self._last_mono = now_mono
        else:
            dt = max(1e-3, now_mono - (self._last_mono or now_mono))
            self._last_mono = now_mono

            err = t_mesh - self._t_disp

            # wenn wir *sehr* weit weg sind (z.B. Bootstrap/Restart), sofort nachziehen
            if abs(err) > snap_if_far_s:
                self._t_disp = t_mesh
            else:
                max_step = (max_slew_ms_per_s / 1000.0) * dt
                self._t_disp += _clamp(err, -max_step, +max_step)

        phase = self._t_disp % period
        return phase < pulse_s


class DummyLED:
    def __init__(self, pin: int):
        self.pin = pin
        self._core = _BlinkCore()
        self._last_print = 0.0

    def update(self, t_mesh: float, period: float = 1.0):
        on = self._core.compute_on(t_mesh, period=period)

        now = time.time()
        if on and not self._core._last_on and (now - self._last_print) > 0.1:
            print(f"[LED pin {self.pin}] BLINK at mesh_time={t_mesh:.3f}")
            self._last_print = now

        self._core._last_on = on


class GrovePiLED:
    def __init__(self, pin: int):
        self.pin = int(pin)
        self._core = _BlinkCore()
        self._last_state = None

        if not HAS_GROVEPI:
            raise RuntimeError("grovepi module not available")

        grovepi.pinMode(self.pin, "OUTPUT")
        grovepi.digitalWrite(self.pin, 0)
        print(f"[GrovePiLED] init on D{self.pin}")

    def update(self, t_mesh: float, period: float = 1.0):
        on = self._core.compute_on(t_mesh, period=period)

        if on != self._last_state:
            try:
                grovepi.digitalWrite(self.pin, 1 if on else 0)
            except IOError:
                print(f"[GrovePiLED] Error writing to D{self.pin}")
            self._last_state = on
