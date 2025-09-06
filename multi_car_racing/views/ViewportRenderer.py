from dataclasses import dataclass

import numpy as np
import pygame


@dataclass
class ViewportRenderer:
    area: pygame.Rect  # gdzie rysować (x, y, w, h)

    def blit_env(self, screen: pygame.Surface, frame_rgb: np.ndarray) -> None:
        surf = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        surf = pygame.transform.smoothscale(surf, (self.area.width, self.area.height))
        screen.blit(surf, self.area.topleft)
