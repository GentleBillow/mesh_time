# mesh/sensor.py

import math
import time
from typing import Optional


class DummySensor:
    """
    Simple simulated sensor:
    returns a slow sine wave + small noise.
    Useful for testing the data path & plotting.
    """

    def __init__(self, sensor_type: str = "dummy"):
        self.sensor_type = sensor_type
        self._start = time.monotonic()

    def read(self) -> float:
        t = time.monotonic() - self._start
        # slow sine wave, period ~60s
        value = 10.0 + 5.0 * math.sin(2 * math.pi * t / 60.0)
        return value
