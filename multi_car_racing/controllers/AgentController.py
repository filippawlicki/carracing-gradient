from typing import Optional

import numpy as np

from .AgentPolicy import AgentPolicy


class AgentController:

    def __init__(self, policy: AgentPolicy):
        self.policy = policy

    def action(self, dt: float, obs: Optional[np.ndarray] = None) -> np.ndarray:
        if obs is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        act = self.policy.act(obs)
        act = np.asarray(act, dtype=np.float32)
        if act.shape != (3,):
            raise ValueError("AgentPolicy.act musi zwracać wektor (3,)")
        return act
