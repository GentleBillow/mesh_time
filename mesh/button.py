# -*- coding: utf-8 -*-
# mesh/button.py

import platform
import time
from typing import Optional, Callable

try:
    import grovepi
    HAS_GROVEPI = True
except ImportError:
    HAS_GROVEPI = False

IS_WINDOWS = platform.system() == "Windows"


class DummyButton:
    """Dummy button for testing (simulates press every 30s)."""
    
    def __init__(self, pin: int):
        self.pin = pin
        self._last_sim_press = 0.0
        self._sim_interval = 30.0  # Simulate press every 30s
        print(f"[DummyButton] init on pin {self.pin} (auto-press every {self._sim_interval}s)")
    
    def read(self) -> bool:
        """Returns True if button is pressed (simulated periodically)."""
        now = time.time()
        if now - self._last_sim_press >= self._sim_interval:
            self._last_sim_press = now
            print(f"[DummyButton pin {self.pin}] Simulated press!")
            return True
        return False


class GrovePiButton:
    """
    Grove Pi button with debouncing.
    Edge-triggered: detects press events (not just state).
    """
    
    def __init__(self, pin: int, debounce_s: float = 0.2):
        self.pin = int(pin)
        self.debounce_s = float(debounce_s)
        self._last_state = 0
        self._last_press_time = 0.0
        
        if not HAS_GROVEPI:
            raise RuntimeError("grovepi module not available")
        
        grovepi.pinMode(self.pin, "INPUT")
        print(f"[GrovePiButton] init on D{self.pin} (debounce={debounce_s}s)")
    
    def read(self) -> bool:
        """
        Returns True on button press event (edge-triggered with debounce).
        """
        try:
            current = grovepi.digitalRead(self.pin)
        except IOError:
            print(f"[GrovePiButton] Error reading from D{self.pin}")
            return False
        
        now = time.time()
        
        # Edge detection: 0 -> 1 transition (button pressed)
        if current == 1 and self._last_state == 0:
            # Debounce check
            if now - self._last_press_time >= self.debounce_s:
                self._last_press_time = now
                self._last_state = current
                print(f"[GrovePiButton D{self.pin}] Press detected!")
                return True
        
        self._last_state = current
        return False
