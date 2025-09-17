import os
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()
        c, h, w = obs_shape
        flat = c * h * w
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 512),
            nn.ReLU(),
        )
        self.mu = nn.Linear(512, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = self.fc(x)
        return self.mu(x), self.log_std, self.value(x)


class PPOController:
    def __init__(self, obs_shape=(3,96,96), action_dim=3,
                 lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4,
                 training: bool = False, save_path: str = "/"):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(obs_shape, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.old_policy = ActorCritic(obs_shape, action_dim)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

        self.memory = {"states": [], "actions": [], "logprobs": [],
                       "rewards": [], "is_terminals": []}

        self.training = training
        self.save_path = save_path
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_act: Optional[np.ndarray] = None

    def _select_action(self, state: np.ndarray):
        state_t = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            mu, log_std, _ = self.old_policy(state_t)
        dist = Normal(mu, log_std.exp())
        action = dist.sample()
        logprob = dist.log_prob(action).sum(axis=-1)
        self.memory["states"].append(state_t.squeeze(0))
        self.memory["actions"].append(action.squeeze(0))
        self.memory["logprobs"].append(logprob)
        return action.squeeze(0).numpy()

    def _store_reward(self, reward, done):
        self.memory["rewards"].append(reward)
        self.memory["is_terminals"].append(done)

    def _update(self):
        rewards = []
        discounted = 0
        for r, done in zip(reversed(self.memory["rewards"]),
                           reversed(self.memory["is_terminals"])):
            if done:
                discounted = 0
            discounted = r + self.gamma * discounted
            rewards.insert(0, discounted)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

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
            loss = -torch.min(surr1, surr2) + \
                   0.5 * self.MseLoss(state_values.squeeze(), rewards) - \
                   0.01 * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.old_policy.load_state_dict(self.policy.state_dict())
        for k in self.memory:
            self.memory[k] = []

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str):
        if os.path.exists(path):
            self.policy.load_state_dict(torch.load(path))
            self.old_policy.load_state_dict(self.policy.state_dict())

    # ---------------- Game adapter -----------------

    def begin_episode(self, obs: Optional[np.ndarray]) -> None:
        self._prev_obs = None
        self._prev_act = None

    def action(self, dt: float, obs: Optional[np.ndarray] = None) -> np.ndarray:
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
        self._store_reward(reward, done=terminated or truncated)
        if terminated or truncated:
            self._update()
            try:
                self.save(self.save_path)
            except Exception:
                pass
            self._prev_obs = None
            self._prev_act = None
