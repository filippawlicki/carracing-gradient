import pygame
import numpy as np


class HumanController:
    def __init__(self, steer_scale: float = 1.0, brake_val: float = 0.8):
        self.steer_scale = steer_scale
        self.brake_val = brake_val

    def action(self, dt: float) -> np.ndarray:
        keys = pygame.key.get_pressed()
        steer = 0.0
        gas = 0.0
        brake = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steer -= 1.0 * self.steer_scale
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steer += 1.0 * self.steer_scale
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            gas = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            brake = self.brake_val
        return np.array([steer, gas, brake], dtype=np.float32)
