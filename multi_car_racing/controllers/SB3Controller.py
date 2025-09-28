import numpy as np
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym


class SB3Controller:
    def __init__(self, training: bool = False, save_path: str = "./trained/sb3_ppo.zip", log: bool = True):
        self.training = training
        self.save_path = save_path
        self.log = log
        self._episode_done = False
        self.episode_count = 0
        self.episode_reward = 0.0
        self._prev_obs = None

        if training:
            # Create a dummy environment for SB3 model initialization
            env = gym.make("CarRacing-v3", render_mode=None)

            # Initialize SB3 PPO model
            self.model = PPO(
                "CnnPolicy",
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1 if log else 0
            )
            env.close()
        else:
            self.model = None


    def begin_episode(self, obs: Optional[np.ndarray]) -> None:
        self._prev_obs = None
        self.episode_reward = 0.0
        self.episode_count += 1
        self._episode_done = False
        if self.log:
            print(f"[SB3] Starting episode {self.episode_count}")

    def action(self, dt: float, obs: Optional[np.ndarray] = None) -> np.ndarray:
        if obs is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if self.model is not None:
            try:
                if obs.shape == (96, 96, 3):
                    obs_processed = obs.transpose(2, 0, 1)
                else:
                    obs_processed = obs

                action, _ = self.model.predict(obs_processed, deterministic=not self.training)

                if isinstance(action, np.ndarray):
                    action = action.astype(np.float32)
                    if action.shape == (3,):
                        self._prev_obs = obs
                        return action
                    elif len(action.shape) == 1 and action.shape[0] == 3:
                        self._prev_obs = obs
                        return action
                    elif action.shape == (): 
                        action = np.array([action, 0.0, 0.0], dtype=np.float32)
                        self._prev_obs = obs
                        return action

            except Exception as e:
                pass

        # Fallback: no action
        action = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._prev_obs = obs
        return action

    def observe(self, reward: float, next_obs: np.ndarray, truncated: bool, terminated: bool) -> None:
        if not self.training:
            return

        self.episode_reward += reward

        if terminated or truncated:
            try:
                if self.model is not None:
                    self.model.save(self.save_path)
            except Exception as e:
                if self.log:
                    print(f"[SB3] Warning: could not save model: {e}")

            self._prev_obs = None
            self._episode_done = True

            if self.log:
                print(f"[SB3] Episode {self.episode_count} finished. Total reward: {self.episode_reward:.2f}")

    def training_done(self) -> bool:
        return False 

    def train_sb3(self, total_timesteps: int = 100000):
        if not self.training or self.model is None:
            print("[SB3] Not in training mode or model not initialized")
            return

        # Create training environment
        env = gym.make("CarRacing-v3", render_mode=None)

        if self.log:
            print(f"[SB3] Starting training for {total_timesteps} timesteps")

        # Train the model
        self.model.set_env(env)
        self.model.learn(total_timesteps=total_timesteps)

        # Save the trained model
        self.model.save(self.save_path)
        env.close()

        if self.log:
            print(f"[SB3] Training completed. Model saved to {self.save_path}")

    def load(self, path: str):
        try:
            self.model = PPO.load(path)
            if self.log:
                print(f"[SB3] Successfully loaded model from {path}")
        except FileNotFoundError:
            if self.log:
                print(f"[SB3] Model file not found: {path}")
        except Exception as e:
            if self.log:
                print(f"[SB3] Could not load model from {path}: {e}")

    def save(self, path: str):
        if self.model is not None:
            self.model.save(path)
            if self.log:
                print(f"[SB3] Model saved to {path}")