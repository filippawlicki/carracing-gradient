from dataclasses import dataclass
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from multi_car_racing.controllers.AgentController import AgentController


@dataclass
class EnvCar:
    env: gym.Env
    last_obs: Optional[np.ndarray] = None
    total_reward: float = 0.0

    def __init__(self, env, controller, role='AI'):
        self.env = env
        self.controller = controller
        self.role = role

    def reset(self, seed: Optional[int] = None) -> None:
        obs, _ = self.env.reset(seed=seed)
        self.last_obs = obs
        self.total_reward = 0.0

    def step(self, dt: float) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if isinstance(self.controller, AgentController):
            action = self.controller.action(dt, self.last_obs)
        else:
            action = self.controller.action(dt)

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.last_obs = obs
        self.total_reward += float(reward)
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render_array(self) -> np.ndarray:
        frame = self.env.render()
        return frame

    def car_xy(self) -> Tuple[float, float]:
        car = getattr(self.env.unwrapped, "car", None)
        if car is None or car.hull is None:
            return (0.0, 0.0)
        return float(car.hull.position.x), float(car.hull.position.y)

    def close(self) -> None:
        self.env.close()
