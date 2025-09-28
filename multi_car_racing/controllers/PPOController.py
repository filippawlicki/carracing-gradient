import os
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        c, h, w = obs_shape

        # Convolutional encoder
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )

        # Compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out_size = self.conv(dummy).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )

        # Actor head
        self.mu = nn.Linear(512, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        """
        x: torch tensor of shape (B, C, H, W), normalized to [0,1]
        returns: mu, log_std, value
        """
        x = self.conv(x)
        x = self.fc(x)
        return self.mu(x), self.log_std, self.value(x)


class PPOController:
    """
    PPO controller directly pluggable into Game.
    Training hyperparameters are passed in `train_config`.
    """
    def __init__(self,
                 obs_shape=(3, 96, 96),
                 action_dim=3,
                 training: bool = False,
                 save_path: str = "./trained/ppo_last.pt",
                 train_config: Optional[Dict[str, Any]] = None,
                 log: bool = True):
        # Default training config
        default_config = {
            "lr": 3e-4,
            "gamma": 0.99,
            "eps_clip": 0.2,
            "k_epochs": 4,
            "max_episodes": 50,
        }
        if train_config:
            default_config.update(train_config)

        self.lr = default_config["lr"]
        self.gamma = default_config["gamma"]
        self.eps_clip = default_config["eps_clip"]
        self.k_epochs = default_config["k_epochs"]
        self.max_episodes = default_config["max_episodes"]

        # Networks and optimizer
        self.policy = ActorCritic(obs_shape, action_dim)
        self.old_policy = ActorCritic(obs_shape, action_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.MseLoss = nn.MSELoss()

        self.memory = {"states": [], "actions": [], "logprobs": [],
                       "rewards": [], "is_terminals": []}

        # Adapter state
        self.training = training
        self.save_path = save_path
        self._episode_done = False
        self.log = log
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_act: Optional[np.ndarray] = None
        self.episode_count = 0
        self.episode_reward = 0.0

        if self.log:
            print(f"[PPO] Initialized with config: {default_config}")

    # Internal PPO methods

    def _select_action(self, state: np.ndarray):
        """Sample action from current policy and store log probability."""
        """
        # Debug: display observation
        if not hasattr(self, '_select_count'):
            self._select_count = 0
        self._select_count += 1

        if self._select_count % 50 == 0:
            print(f"[DEBUG] BEFORE neural network - state shape: {state.shape}, range: [{state.min()}, {state.max()}]")
            plt.figure(figsize=(8, 4))

            plt.subplot(1, 2, 1)
            plt.imshow(state)
            plt.title(f"Original state (before processing)")
            plt.axis('off')

            plt.tight_layout()
            plt.show(block=False)
            plt.pause(10.0)
            plt.close()
        """
        state_t = torch.FloatTensor(state / 255.0).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            mu, log_std, _ = self.old_policy(state_t)
        dist = Normal(mu, log_std.exp())
        raw_action = dist.sample()

        # Scale to CarRacing ranges
        steer = torch.tanh(raw_action[:, 0])  # [-1,1]
        gas = torch.sigmoid(raw_action[:, 1])  # [0,1]
        brake = torch.sigmoid(raw_action[:, 2])  # [0,1]

        action = torch.stack([steer, gas, brake], dim=-1)
        logprob = dist.log_prob(raw_action).sum(axis=-1)

        # Store memory
        self.memory["states"].append(state_t.squeeze(0))
        self.memory["actions"].append(raw_action.squeeze(0))
        self.memory["logprobs"].append(logprob)

        return action.squeeze(0).numpy()

    def _store_reward(self, reward, done):
        self.memory["rewards"].append(reward)
        self.memory["is_terminals"].append(done)

    def _update(self):
        """Perform PPO update using collected trajectory."""
        # Compute discounted rewards
        rewards = []
        discounted = 0
        for r, done in zip(reversed(self.memory["rewards"]),
                           reversed(self.memory["is_terminals"])):
            if done:
                discounted = 0
            discounted = r + self.gamma * discounted
            rewards.insert(0, discounted)

        rewards = torch.tensor(rewards, dtype=torch.float32)
        if rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-7)
        else:
            rewards = rewards - rewards.mean()

        old_states = torch.stack(self.memory["states"])
        old_actions = torch.stack(self.memory["actions"]).detach()
        old_logprobs = torch.stack(self.memory["logprobs"]).detach()

        for _ in range(self.k_epochs):
            mu, log_std, state_values = self.policy(old_states)
            dist = Normal(mu, log_std.exp())
            logprobs = dist.log_prob(old_actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1)
            ratios = torch.exp(logprobs - old_logprobs)

            advantages = rewards - state_values.squeeze().detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            state_values = state_values.squeeze(-1)  # keep batch dimension
            loss = -torch.min(surr1, surr2) + \
                   0.5 * self.MseLoss(state_values, rewards) - \
                   0.01 * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        for k in self.memory:
            self.memory[k] = []

        if self.log:
            print(f"[PPO] Policy updated at episode {self.episode_count}")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)
        if self.log:
            print(f"[PPO] Policy saved to {path}")

    def load(self, path: str):
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path))
            self.old_policy.load_state_dict(self.policy.state_dict())
            if self.log:
                print(f"[PPO] Loaded policy from {path}")

    # Game adapter

    def begin_episode(self, obs: Optional[np.ndarray]) -> None:
        self._prev_obs = None
        self._prev_act = None
        self.episode_reward = 0.0
        self.episode_count += 1
        self._episode_done = False
        if self.log:
            print(f"[PPO] Starting episode {self.episode_count}/{self.max_episodes}")


    def action(self, dt: float, obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Return action for the current observation."""
        if obs is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
        act = np.asarray(self._select_action(obs), dtype=np.float32)
        if act.shape != (3,):
            raise ValueError("action() must return vector (3,)")
        self._prev_obs = obs
        self._prev_act = act
        return act

    def observe(self, reward: float, next_obs: np.ndarray, truncated: bool, terminated: bool) -> None:
        if not self.training:
            return

        if self._prev_obs is None or self._prev_act is None:
            self._prev_obs = next_obs
            return

        self.episode_reward += reward
        self._store_reward(reward, done=terminated or truncated)

        if terminated or truncated:
            self._update()
            try:
                self.save(self.save_path)
            except Exception as e:
                if self.log:
                    print(f"[PPO] Warning: could not save policy: {e}")

            self._prev_obs = None
            self._prev_act = None
            self._episode_done = True  # mark episode done

            if self.log:
                print(f"[PPO] Episode {self.episode_count} finished. Total reward: {self.episode_reward:.2f}")

    def training_done(self) -> bool:
        return self.episode_count >= self.max_episodes
