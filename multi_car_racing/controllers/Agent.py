from typing import Protocol

import numpy as np


class Agent(Protocol):
    def action(self, obs: np.ndarray) -> np.ndarray:
        ...

    def learn(self,
              obs: np.ndarray,
              next_obs: np.ndarray,
              action: np.ndarray,
              reward: float,
              terminated: bool,
              truncated: bool) -> None:
        ...

    def save(self, path: str):
        ...

    def load(self, path: str):
        ...
