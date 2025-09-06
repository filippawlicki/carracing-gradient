from typing import Protocol

import numpy as np


class AgentPolicy(Protocol):
    def act(self, obs: np.ndarray) -> np.ndarray:
        ...
