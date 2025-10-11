from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np


@dataclass
class TrackMap:
    size: int = 200
    margin: int = 50

    seed: int = 0
    bounds: Optional[Tuple[float, float, float, float]] = None  # (min_x, max_x, min_y, max_y)
    scale: Optional[float] = None
    offset_xy: Optional[Tuple[float, float]] = None
    image: Optional[np.ndarray] = None  # (size,size,3) uint8

    def build_from_env(self, env: gym.Env) -> None:
        env.reset(seed=self.seed)
        track_nodes = env.unwrapped.track
        if not track_nodes:
            self.image = np.zeros((self.size, self.size, 3), dtype=np.uint8)
            self.bounds = (0.0, 1.0, 0.0, 1.0)
            self.scale = 1.0
            self.offset_xy = (0.0, 0.0)
            return

        x_coords = [node[2] for node in track_nodes]
        y_coords = [node[3] for node in track_nodes]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        min_x -= self.margin
        max_x += self.margin
        min_y -= self.margin
        max_y += self.margin

        width = max_x - min_x
        height = max_y - min_y
        scale = min(self.size / width, self.size / height) * 0.9
        off_x = (self.size - width * scale) / 2
        off_y = (self.size - height * scale) / 2

        minimap = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        for i in range(len(track_nodes)):
            curr = track_nodes[i]
            nxt = track_nodes[(i + 1) % len(track_nodes)]
            x1 = int((curr[2] - min_x) * scale + off_x)
            y1 = int((curr[3] - min_y) * scale + off_y)
            x2 = int((nxt[2] - min_x) * scale + off_x)
            y2 = int((nxt[3] - min_y) * scale + off_y)
            cv2.line(minimap, (x1, y1), (x2, y2), (255, 255, 255), 3)

        self.image = minimap
        self.bounds = (min_x, max_x, min_y, max_y)
        self.scale = scale
        self.offset_xy = (off_x, off_y)

    def world_to_minimap(self, x: float, y: float) -> Tuple[int, int]:
        if self.bounds is None or self.scale is None or self.offset_xy is None:
            return self.size // 2, self.size // 2
        min_x, max_x, min_y, max_y = self.bounds
        scale = self.scale
        off_x, off_y = self.offset_xy
        mx = int((x - min_x) * scale + off_x)
        my = int((y - min_y) * scale + off_y)
        mx = max(0, min(self.size - 1, mx))
        my = max(0, min(self.size - 1, my))
        return mx, my
