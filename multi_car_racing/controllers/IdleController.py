import numpy as np


class IdleController:

    def action(self, dt: float) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
