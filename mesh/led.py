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


class DummyLED:
    """
    Dev-/Fallback-LED: druckt nur BLINK-Events.
    """

    def __init__(self, pin: int):
        self.pin = pin
        self._last_state = False
        self._last_print = 0.0

    def update(self, t_mesh: float, period: float = 1.0, epsilon: float = 0.02):
        """
        Blink wenn mesh_time ein Vielfaches von 'period' schneidet.
        """
        phase = t_mesh % period
        on = phase < epsilon

        now = time.time()
        if on and not self._last_state and (now - self._last_print) > 0.1:
            print(f"[LED pin {self.pin}] BLINK at mesh_time={t_mesh:.3f}")
            self._last_print = now

        self._last_state = on


class GrovePiLED:
    """
    LED auf einem GrovePi/Grove Base HAT.

    pin = Grove-Digital-Port (z.B. 3 für D3, 4 für D4, ...)
    """

    def __init__(self, pin: int):
        self.pin = int(pin)
        self._last_state = None

        if not HAS_GROVEPI:
            raise RuntimeError("grovepi module not available")

        # GrovePi-Pin initialisieren
        grovepi.pinMode(self.pin, "OUTPUT")
        grovepi.digitalWrite(self.pin, 0)
        print(f"[GrovePiLED] init on D{self.pin}")

    def update(self, t_mesh: float, period: float = 1.0, epsilon: float = 0.02):
        """
        Blink-Schema wie DummyLED, aber echtes digitalWrite.
        """
        phase = t_mesh % period
        on = phase < epsilon

        # nur schreiben, wenn sich der Zustand ändert
        if on != self._last_state:
            try:
                grovepi.digitalWrite(self.pin, 1 if on else 0)
            except IOError:
                print(f"[GrovePiLED] Error writing to D{self.pin}")
            self._last_state = on
