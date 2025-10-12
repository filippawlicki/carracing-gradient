import random
from typing import Optional

import numpy as np
import pygame
import gymnasium as gym

from .controllers.HumanController import HumanController
from .controllers.PPOController import PPOController
from .views.TrackMap import TrackMap
from .views.MiniMap import MiniMap
from .envs.EnvCar import EnvCar
from .views.ViewportRenderer import ViewportRenderer
from .GameConfig import GameConfig


class Game:
    def __init__(self, config: GameConfig):
        self.config = config

        if not config.render and self.config.human:
            raise ValueError("Human cannot be applied while rendering is off")

        #results = []
#
        #for i in range(1000):
        #    self.seed = self.config.seed if self.config.seed is not None else int(np.random.randint(0, 10_000))
        #    env = gym.make("CarRacing-v3", render_mode=None)
        #    self.track_map = TrackMap(seed=self.seed, size=config.minimap_size)
#
        #    try:
        #        self.track_map.build_from_env(env)
        #    finally:
        #        env.close()
#
        #    curvature = self._compute_curvature(self.track_map)
        #    results.append((self.seed, curvature, self.track_map))
#
        #results.sort(key=lambda x: x[1])
#
        #best_10 = results[:10]
        #for seed, curvature, track_map in best_10:
        #    print(f"✔ Seed {seed} — curvature={curvature:.4f}")
        #    track_map.save_map()
#
        #for seed, roundness, track_map in best_10:
        #    print(f"✔ Seed {seed} — roundness={roundness:.3f}")
        #    track_map.save_map()

        self.seed = self.config.seed if self.config.seed is not None else int(np.random.randint(0, 10_000))

        self.seed = random.choice([495, 1678, 1906, 2836, 8400, 5755, 4293])

        self.clock = pygame.time.Clock()
        self.screen: Optional[pygame.Surface] = None

        self.cars = []

        self.track_map = TrackMap(seed=self.seed, size=config.minimap_size)

        self.minimap = MiniMap(track=self.track_map)

        self.viewports: list[ViewportRenderer] = []

        self.font_title: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None



    def init_pygame(self) -> None:
        if not self.config.render:
            return
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        pygame.display.set_caption("CarRacing – Dual View (OOP)")
        self.font_title = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 22)

        n = self.config.number_of_cars
        view_w = (self.config.width - self.config.minimap_size - (10 * (n + 1))) // n
        view_h = self.config.height - 100

        self.viewports = []
        for i in range(n):
            x = 10 + i * (view_w + 10)
            y = 50
            vp = ViewportRenderer(pygame.Rect(x, y, view_w, view_h))
            self.viewports.append(vp)

    def _make_env(self) -> gym.Env:
        render_mode = None if not self.config.render else "rgb_array"
        return gym.make("CarRacing-v3", render_mode=render_mode, max_episode_steps=1500)

    def init_envs(self) -> None:
        base_env = self._make_env()
        base_env.reset(seed=self.seed)

        self.track_map.build_from_env(base_env)

        for i in range(self.config.number_of_cars):
            env = self._make_env()
            env.reset(seed=self.seed)

            if self.config.human and i == 0:
                controller = HumanController()
                role = "PLAYER"
            else:
                controller = self.config.user_agent_factory()
                role = "AI"

            car = EnvCar(env=env, controller=controller, role=role)
            self.cars.append(car)

    def reset_both(self) -> None:
        for car in self.cars:
            assert car
            car.reset(self.seed)
            if car is self.cars[-1]:
                self.track_map.build_from_env(car.env)

    def draw_headers(self) -> None:
        assert self.screen and self.font_title

        for car, vp in zip(self.cars, self.viewports):
            color = (0, 255, 0) if car.role == "PLAYER" else (255, 100, 100)
            title = self.font_title.render(car.role, True, color)
            rect = title.get_rect(center=(vp.area.centerx, 25))
            pygame.draw.rect(self.screen, (0, 0, 0), rect.inflate(20, 10))
            self.screen.blit(title, rect)

    def draw_minimap(self) -> None:
        assert self.screen

        x = self.config.width - self.config.minimap_size - 10
        y = 10
        surf = self.minimap.to_surface()
        # tło + mapa
        pygame.draw.rect(self.screen, (0, 0, 0), pygame.Rect(x, y, self.config.minimap_size, self.config.minimap_size))
        self.screen.blit(surf, (x, y))

        players = []
        for car in self.cars:
            assert car
            car_x, car_y = car.car_xy()
            player = self.track_map.world_to_minimap(car_x, car_y)
            players.append(player)

        self.minimap.draw_cars(self.screen, (x, y), players)

    def draw_instructions(self) -> None:
        assert self.screen and self.font_small
        tips = [
            "WASD/Arrows - Controls",
            "R – Reset",
            "ESC – Quit",
            "Minimap on the top-right (green dot is you)",
        ]
        for i, line in enumerate(tips):
            text = self.font_small.render(line, True, (200, 200, 200))
            rect = text.get_rect(bottomright=(self.config.width - 10, self.config.height - 10 - i * 25))
            self.screen.blit(text, rect)

    def handle_events(self) -> bool:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    return False
                if e.key == pygame.K_r:
                    self.reset_both()
        return True

    def is_running(self, truncated, terminated) -> bool:
        return not (terminated or truncated)

    def run(self) -> None:
        if self.config.render:
            self.init_pygame()
            assert self.screen
        self.init_envs()

        for car in self.cars:
            assert car

        running = True
        while running:
            if self.config.render:
                dt = self.clock.tick(self.config.fps) / 1000.0
            else:
                dt = 0  # arbitrary small timestep, only needed for controllers

            if self.config.render:
                self.screen.fill((0, 0, 0))
                self.draw_headers()

            step_results = []
            if self.config.render:
                for car, vp in zip(self.cars, self.viewports):
                    obs, reward, terminated, truncated, _ = car.step(dt)
                    step_results.append((car, obs, reward, terminated, truncated))
                    frame = car.render_array()
                    vp.blit_env(self.screen, frame)
            else:
                for car in self.cars:
                    obs, reward, terminated, truncated, _ = car.step(dt)
                    step_results.append((car, obs, reward, terminated, truncated))

            if self.config.user_agent_training:
                for car, obs, reward, terminated, truncated in step_results:
                    if hasattr(car.controller, "observe"):
                        car.controller.observe(next_obs=obs, reward=reward, terminated=terminated, truncated=truncated)
                        if car.controller._episode_done:
                            car.reset(self.seed)

            if self.config.render:
                self.draw_minimap()
                self.draw_instructions()
                pygame.display.flip()
                running = running and self.handle_events()

            if self.config.user_agent_training:
                all_done = all(car.controller.training_done() for car in self.cars)
                if all_done:
                    if self.config.render and self.log:
                        print("[Game] All agents finished training. Exiting.")
                    break

        for car in self.cars:
            car.close()

        if self.config.render:
            pygame.quit()



    #method for generating and saving maps
    ##def _compute_curvature(self, track_map: 'TrackMap') -> float:
    ##    """
    ##    Oblicza średnią zmianę kąta między kolejnymi odcinkami toru.
    ##    Im mniejsza wartość, tym tor ma łagodniejsze zakręty.
    ##    """
    ##    env = gym.make("CarRacing-v3", render_mode=None)
    ##    env.reset(seed=track_map.seed)
    ##    nodes = env.unwrapped.track
    ##    env.close()
##
    ##    if not nodes or len(nodes) < 3:
    ##        return float('inf')
##
    ##    angles = []
    ##    for i in range(len(nodes)):
    ##        x1, y1 = nodes[i][2], nodes[i][3]
    ##        x2, y2 = nodes[(i + 1) % len(nodes)][2], nodes[(i + 1) % len(nodes)][3]
    ##        x3, y3 = nodes[(i + 2) % len(nodes)][2], nodes[(i + 2) % len(nodes)][3]
##
    ##        # wektory odcinków
    ##        v1 = np.array([x2 - x1, y2 - y1])
    ##        v2 = np.array([x3 - x2, y3 - y2])
##
    ##        # kąt między nimi
    ##        dot = np.dot(v1, v2)
    ##        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    ##        if norm == 0:
    ##            continue
    ##        cos_angle = np.clip(dot / norm, -1.0, 1.0)
    ##        angle = np.arccos(cos_angle)
    ##        angles.append(angle)
##
    ##    if not angles:
    ##        return float('inf')
##
    ##    # średnia zmiana kąta (radiany)
    ##    return float(np.mean(angles))
