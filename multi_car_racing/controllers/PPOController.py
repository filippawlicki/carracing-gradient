import os
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        c, h, w = obs_shape

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

        # Actor head - separate outputs for each action
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
    def __init__(self,
                 obs_shape=(3, 96, 96),
                 action_dim=3,
                 training: bool = False,
                 save_path: str = "./trained/ppo_last.pt",
                 train_config: Optional[Dict[str, Any]] = None,
                 log: bool = True):
        self.log = log

        # Default training config
        default_config = {
            "lr": 3e-4,
            "gamma": 0.99,
            "eps_clip": 0.2,
            "k_epochs": 4,
            "max_episodes": 100,
            "value_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
        }
        if train_config:
            default_config.update(train_config)

        self.lr = default_config["lr"]
        self.gamma = default_config["gamma"]
        self.eps_clip = default_config["eps_clip"]
        self.k_epochs = default_config["k_epochs"]
        self.max_episodes = default_config["max_episodes"]
        self.value_coef = default_config["value_coef"]
        self.entropy_coef = default_config["entropy_coef"]
        self.max_grad_norm = default_config["max_grad_norm"]

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            if self.log:
                print("Using MPS")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.log:
                print("Using CUDA")
        else:
            self.device = torch.device("cpu")
            if self.log:
                print("Using CPU")

        # Networks and optimizer
        self.policy = ActorCritic(obs_shape, action_dim).to(self.device)
        self.old_policy = ActorCritic(obs_shape, action_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.MseLoss = nn.MSELoss()

        self.memory = {"states": [], "actions": [], "logprobs": [],
                       "rewards": [], "is_terminals": [], "values": []}

        # Adapter state
        self.training = training
        self.save_path = save_path
        self._episode_done = False
        self.log = log
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_act: Optional[np.ndarray] = None
        self.episode_count = 0
        self.episode_reward = 0.0
        self.total_steps = 0

        if self.log:
            print(f"Initialized with config: {default_config}")

    def _select_action(self, state: np.ndarray):
        # Normalize to [0, 1] and convert to tensor
        state_t = torch.FloatTensor(state / 255.0).permute(2, 0, 1).unsqueeze(0)
        state_t = state_t.to(self.device)

        with torch.no_grad():
            mu, log_std, value = self.old_policy(state_t)

        dist = Normal(mu, log_std.exp())
        raw_action = dist.sample()
        logprob = dist.log_prob(raw_action).sum(axis=-1)

        # Store raw action and logprob
        self.memory["states"].append(state_t.squeeze(0))
        self.memory["actions"].append(raw_action.squeeze(0))
        self.memory["logprobs"].append(logprob.squeeze(0))
        self.memory["values"].append(value.squeeze(0))

        # Convert to CarRacing action space
        steer = torch.tanh(raw_action[:, 0])
        gas = torch.sigmoid(raw_action[:, 1])
        brake = torch.sigmoid(raw_action[:, 2])

        action = torch.stack([steer, gas, brake], dim=-1)
        return action.squeeze(0).cpu().numpy()

    def _store_reward(self, reward, done):
        self.memory["rewards"].append(reward)
        self.memory["is_terminals"].append(done)

    def _update(self):
        if len(self.memory["rewards"]) == 0:
            return

        # Compute GAE
        rewards = []
        discounted = 0
        for r, done in zip(reversed(self.memory["rewards"]),
                           reversed(self.memory["is_terminals"])):
            if done:
                discounted = 0
            discounted = r + self.gamma * discounted
            rewards.insert(0, discounted)

        rewards = torch.tensor(rewards, dtype=torch.float32)

        # Normalize rewards
        if rewards.numel() > 1:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        old_states = torch.stack(self.memory["states"])
        old_actions = torch.stack(self.memory["actions"]).detach()
        old_logprobs = torch.stack(self.memory["logprobs"]).detach()
        old_values = torch.stack(self.memory["values"]).detach()

        rewards = rewards.to(old_values.device)

        for epoch in range(self.k_epochs):
            mu, log_std, state_values = self.policy(old_states)
            dist = Normal(mu, log_std.exp())
            logprobs = dist.log_prob(old_actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()

            ratios = torch.exp(logprobs - old_logprobs)

            # Compute advantages
            advantages = rewards - old_values.squeeze()

            # Actor loss with clipping
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            critic_loss = self.MseLoss(state_values.squeeze(), rewards)

            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            # Gradient update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())

        # Clear memory
        for k in self.memory:
            self.memory[k] = []

        if self.log:
            print(f"[PPO] Policy updated at episode {self.episode_count} "
                  f"(actor_loss: {actor_loss.item():.4f}, "
                  f"critic_loss: {critic_loss.item():.4f}, "
                  f"entropy: {entropy.item():.4f})")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
        }, path)
        if self.log:
            print(f"Policy saved to {path}")

    def load(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.old_policy.load_state_dict(self.policy.state_dict())
            if 'episode_count' in checkpoint:
                self.episode_count = checkpoint['episode_count']
            if self.log:
                print(f"Loaded policy from {path} (episode {self.episode_count})")
        else:
            if self.log:
                print(f"Warning: No checkpoint found at {path}, starting fresh")

    def begin_episode(self, obs: Optional[np.ndarray]) -> None:
        self._prev_obs = None
        self._prev_act = None
        self.episode_reward = 0.0
        self.episode_count += 1
        self._episode_done = False
        if self.log:
            print(f"Starting episode {self.episode_count}/{self.max_episodes}")

    def action(self, dt: float, obs: Optional[np.ndarray] = None) -> np.ndarray:
        if obs is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        act = self._select_action(obs).astype(np.float32)

        if act.shape != (3,):
            raise ValueError(f"action() must return vector (3,), got {act.shape}")

        self._prev_obs = obs
        self._prev_act = act
        self.total_steps += 1

        return act

    def observe(self, reward: float, next_obs: np.ndarray,
                truncated: bool, terminated: bool) -> None:
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
                    print(f"Warning: could not save policy: {e}")

            self._prev_obs = None
            self._prev_act = None
            self._episode_done = True

            if self.log:
                print(f"Episode {self.episode_count} finished. "
                      f"Total reward: {self.episode_reward:.2f}, "
                      f"Total steps: {self.total_steps}")

    def training_done(self) -> bool:
        return self.episode_count >= self.max_episodes