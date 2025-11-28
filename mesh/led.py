# mesh/led.py

import time


class DummyLED:
    """
    LED controller that just prints when it would turn on.
    On a real Pi, this will wrap a GPIO LED instead.
    """

    def __init__(self, pin: int):
        self.pin = pin
        self._last_state = False
        self._last_print = 0.0

    def update(self, t_mesh: float, period: float = 0.5, epsilon: float = 0.01):
        """
        Blink when mesh time crosses multiples of 'period' seconds.
        Here we just print transitions, not a continuous on/off signal.

        epsilon: tolerance around exact multiple.
        """
        phase = t_mesh % period
        on = phase < epsilon  # very simple: "blink" at the start of each period

        now = time.time()
        if on and not self._last_state and (now - self._last_print) > 0.1:
            print(f"[LED pin {self.pin}] BLINK at mesh_time={t_mesh:.3f}")
            self._last_print = now

        self._last_state = on
