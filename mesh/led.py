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
    Edge-triggered global blink.
    No phase slewing. Uses t_mesh as absolute event clock.
    """
    def __init__(self):
        self._last_k = None
        self._pulse_until = 0.0

    def compute_on(
        self,
        t_mesh: float,
        *,
        period: float = 1.0,
        pulse_s: float = 0.05,
    ) -> bool:
        now = time.monotonic()

        k = int(t_mesh // period)

        if self._last_k is None:
            self._last_k = k

        if k != self._last_k:
            # global event edge
            self._last_k = k
            self._pulse_until = now + pulse_s

        return now < self._pulse_until



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
