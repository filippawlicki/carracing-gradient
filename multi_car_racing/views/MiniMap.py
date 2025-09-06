from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pygame

from .TrackMap import TrackMap


@dataclass
class MiniMap:
    track: TrackMap

    def to_surface(self) -> pygame.Surface:
        assert self.track.image is not None, "Najpierw wywołaj TrackMap.build_from_env()"
        # pygame oczekuje (W,H,3) po transpozycji
        return pygame.surfarray.make_surface(np.transpose(self.track.image, (1, 0, 2)))

    def draw_cars(self, screen: pygame.Surface, top_left: Tuple[int, int], cars_px: list[Tuple[int, int]]):
        x0, y0 = top_left
        size = self.track.size
        rect = pygame.Rect(x0, y0, size, size)
        # ramka
        pygame.draw.rect(screen, (255, 255, 255), rect, 2)
        # kropki aut
        colors = [(0, 255, 0), (255, 100, 100), (100, 200, 255), (255, 200, 0)]
        for i, (cx, cy) in enumerate(cars_px):
            col = colors[i % len(colors)]
            pygame.draw.circle(screen, col, (x0 + cx, y0 + cy), 5)