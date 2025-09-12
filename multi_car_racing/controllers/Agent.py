from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np


class Agent(ABC):
    """Minimal interface an agent must implement to be used by AgentController."""

    @abstractmethod
    def action(self, obs: np.ndarray) -> np.ndarray:
        """
        Given current observation (image), return an action vector [steer, gas, brake]
        shape must be (3,) and dtype float32.
        """
        raise NotImplementedError

    @abstractmethod
    def learn(
        self,
        prev_obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Called during training to let the agent learn from a transition."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist agent parameters to disk."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent parameters from disk (if file exists)."""
        raise NotImplementedError
