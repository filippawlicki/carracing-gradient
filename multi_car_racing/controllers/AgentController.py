from typing import Optional
import numpy as np

from multi_car_racing.agents import PPOAgent


class AgentController:
    """Wraps an Agent and provides begin_episode / action / observe hooks used by the Game."""

    def __init__(self, agent: PPOAgent, training: bool = False, save_path: str = "/"):
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
            raise ValueError("Agent.action must return a vector with shape (3,)")
        # save for learning step
        self._prev_obs = obs
        self._prev_act = act
        return act

    def observe(
        self,
        reward: float,
        next_obs: np.ndarray,
        truncated: bool,
        terminated: bool,
    ) -> None:
        """
        Called by the environment loop after each step. If training is enabled,
        pass the transition to the agent.learn method.
        """
        if not self.training:
            return

        # If we don't have a previous obs/action we cannot learn yet.
        if self._prev_obs is None or self._prev_act is None:
            # save the next observation as previous for the next step
            self._prev_obs = next_obs
            return

        # Agent.learn(prev_obs, action, reward, next_obs, terminated, truncated)
        self.agent.learn(
            self._prev_obs, self._prev_act, reward, next_obs, terminated, truncated
        )

        if truncated or terminated:
            # save final policy/params at episode boundary
            try:
                self.agent.save(self.save_path)
            except Exception:
                # do not crash on save failures
                pass
            self._prev_obs = None
            self._prev_act = None
