import os
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
import cv2


class SB3Controller:
    def __init__(self,
                 training: bool = False,
                 save_path: str = "./trained/sb3_ppo.zip",
                 log: bool = True):
        self.training = training
        self.save_path = save_path
        self.log = log
        self.model: Optional[PPO] = None
        self._episode_done = False
        self.episode_count = 0
        self.episode_reward = 0.0
        self._prev_obs: Optional[np.ndarray] = None
        self.expected_shape = None

    def load(self, path: str):
        """Load a pre-trained SB3 PPO model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"[SB3] Model not found at {path}")

        try:
            # Load with custom objects to handle old gym models
            custom_objects = {
                "learning_rate": 0.0,
                "lr_schedule": lambda _: 0.0,
                "clip_range": lambda _: 0.0,
            }
            self.model = PPO.load(path, custom_objects=custom_objects)

            # Get expected observation shape from the model
            if hasattr(self.model.policy, 'observation_space'):
                obs_space = self.model.policy.observation_space
                self.expected_shape = obs_space.shape
                if self.log:
                    print(f"Model expects observation shape: {self.expected_shape}")

            if self.log:
                print(f"Loaded model from {path}")

        except Exception as e:
            if self.log:
                print(f"Error loading model: {e}")
            raise

    def save(self, path: str):
        if self.model is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            if self.log:
                print(f"Model saved to {path}")

    def begin_episode(self, obs: Optional[np.ndarray]) -> None:
        self._prev_obs = obs
        self.episode_reward = 0.0
        self.episode_count += 1
        self._episode_done = False
        if self.log and self.training:
            print(f"Starting episode {self.episode_count}")

    def action(self, dt: float, obs: Optional[np.ndarray] = None) -> np.ndarray:
        if obs is None or self.model is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # SB3 expects (H, W, C) format from gymnasium
        try:
            action, _ = self.model.predict(obs, deterministic=not self.training)
        except ValueError as e:
            error_msg = str(e)
            if "Unexpected observation shape" in error_msg:
                if self.log:
                    print(f"Shape mismatch detected.")
                    print(f"Environment obs shape: {obs.shape}")
                    print(f"Model expects: {self.expected_shape}")
                # Return safe default action
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            raise
        except Exception as e:
            if self.log:
                print(f"Error during prediction: {e}")
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Ensure correct shape and type
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,):
            # Handle different action shapes
            if len(action.shape) == 2 and action.shape[0] == 1:
                action = action[0]  # Take first element if batched
            else:
                if self.log:
                    print(f"Warning: unexpected action shape {action.shape}, using default")
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self._prev_obs = obs
        return action

    def observe(self, reward: float, next_obs: np.ndarray,
                truncated: bool, terminated: bool) -> None:
        """Called after each step. SB3 handles training internally."""
        self.episode_reward += reward

        if terminated or truncated:
            self._episode_done = True
            if self.log:
                print(f"[SB3] Episode {self.episode_count} finished. "
                      f"Total reward: {self.episode_reward:.2f}")

    def training_done(self) -> bool:
        """SB3 training is handled separately, so always return True."""
        return not self.training