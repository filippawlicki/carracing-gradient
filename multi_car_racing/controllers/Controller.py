from typing import Protocol

import numpy as np


class Controller(Protocol):
    def action(self, dt: float) -> np.ndarray:
        """Zwraca akcję [steer, gas, brake]"""
        ...
