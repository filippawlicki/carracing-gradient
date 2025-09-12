import json
import os
from typing import Optional

import numpy as np


class SimpleAgent:
    """
    A lightweight deterministic agent based on color heuristics.
    This is intended as a simple baseline and also supports the Agent interface.
    """

    def __init__(self):
        self.t = 0

        # parameters
        self.base_gas = 0.85
        self.min_gas_on_grass = 0.25
        self.max_brake = 0.6
        self.max_steer = 0.6
        self.wave_freq = 0.12

        # grass detection - be conservative
        self.green_thresh = 1.30
        self.side_gain = 0.30
        self.warmup_steps = 200

    def action(self, obs: np.ndarray) -> np.ndarray:
        """
        obs: (H, W, 3) uint8 image from the environment
        returns: [steer, gas, brake] (shape (3,))
        """
        self.t += 1

        # 1) small steering oscillation to keep moving
        wave = np.sin(self.t * self.wave_freq) * self.max_steer

        # 2) analyze color in a narrow lower-central band
        h, w, _ = obs.shape
        y0, y1 = int(h * 0.55), int(h * 0.90)
        x0, x1 = int(w * 0.35), int(w * 0.65)
        roi = obs[y0:y1, x0:x1, :].astype(np.float32) + 1e-6

        r, g, b = roi[..., 0], roi[..., 1], roi[..., 2]
        green_ratio = g / ((r + b) / 2.0)

        # robust metric via percentile
        gr_metric = float(np.percentile(green_ratio, 85))
        on_grass = gr_metric > self.green_thresh

        # 3) side correction using wider lower field
        roi_side = obs[int(h * 0.55):int(h * 0.90), :, :].astype(np.float32)
        r2, g2, b2 = roi_side[..., 0], roi_side[..., 1], roi_side[..., 2]
        gr2 = g2 / ((r2 + b2) / 2.0)
        left = np.percentile(gr2[:, : w // 2], 80)
        right = np.percentile(gr2[:, w // 2 :], 80)
        side_corr = np.clip((left - right), -1.0, 1.0) * self.side_gain

        steer = np.clip(wave - side_corr, -1.0, 1.0)

        # 4) gas / brake logic
        if self.t <= self.warmup_steps:
            gas = 0.95
            brake = 0.0
        else:
            if on_grass:
                gas = self.min_gas_on_grass
                brake = np.clip((gr_metric - self.green_thresh) * 0.8, 0.0, self.max_brake)
            else:
                gas = self.base_gas
                brake = 0.0

        return np.array([steer, gas, brake], dtype=np.float32)

    # --- learning signature compatible with Agent.learn(prev_obs, action, reward, next_obs, terminated, truncated)
    def learn(
        self,
        prev_obs,
        action,
        reward,
        next_obs,
        terminated: bool,
        truncated: bool,
    ):
        # this simple heuristic agent does not learn; training stub
        return

    def save(self, path: str):
        data = {
            "t": int(self.t),
            "base_gas": float(self.base_gas),
            "min_gas_on_grass": float(self.min_gas_on_grass),
            "max_brake": float(self.max_brake),
            "max_steer": float(self.max_steer),
            "wave_freq": float(self.wave_freq),
            "green_thresh": float(self.green_thresh),
            "side_gain": float(self.side_gain),
            "warmup_steps": int(self.warmup_steps),
        }
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.t = int(data.get("t", 0))
        self.base_gas = float(data.get("base_gas", self.base_gas))
        self.min_gas_on_grass = float(data.get("min_gas_on_grass", self.min_gas_on_grass))
        self.max_brake = float(data.get("max_brake", self.max_brake))
        self.max_steer = float(data.get("max_steer", self.max_steer))
        self.wave_freq = float(data.get("wave_freq", self.wave_freq))
        self.green_thresh = float(data.get("green_thresh", self.green_thresh))
        self.side_gain = float(data.get("side_gain", self.side_gain))
        self.warmup_steps = int(data.get("warmup_steps", self.warmup_steps))
