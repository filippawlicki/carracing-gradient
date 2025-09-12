from typing import Optional

import numpy as np

from .Agent import Agent


class AgentController:

    def __init__(self, agent: Agent, training: bool = False, save_path: str = '/'):
        self.agent = agent
        self.training = training
        self.save_path = save_path
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_act: Optional[np.ndarray] = None

    def begin_episode(self, obs: Optional[np.ndarray]) -> None:
        self._prev_obs = None
        self._prev_act = None

    def action(self, dt: float, obs: Optional[np.ndarray] = None) -> np.ndarray:
        if obs is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        act = np.asarray(self.agent.action(obs), dtype=np.float32)
        if act.shape != (3,):
            raise ValueError("Agent.action musi zwracać wektor (3,)")
        self._prev_obs = obs
        self._prev_act = act
        return act

    def observe(self, reward: float, next_obs: np.ndarray, truncated: bool, terminated: bool) -> None:
        if not self.training:
            return
        if self._prev_obs is None or self._prev_act is None:
            self._prev_obs = next_obs
            return
        self.agent.learn(self._prev_obs, next_obs, self._prev_act, reward, truncated, terminated)
        if truncated or terminated:
            self.agent.save("trained/last.json")
            self._prev_obs = None
            self._prev_act = None
