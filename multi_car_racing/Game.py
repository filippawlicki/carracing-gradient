from typing import Optional

import numpy as np
import pygame
import gymnasium as gym

from .controllers.HumanController import HumanController
from .controllers.AgentController import AgentController
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

        self.seed = self.config.seed if self.config.seed is not None else int(np.random.randint(0, 10_000))

        self.clock = pygame.time.Clock()
        self.screen: Optional[pygame.Surface] = None

        self.cars = []

        self.track_map = TrackMap(size=config.minimap_size)
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
                role = "GRACZ"
            else:
                agent = self.config.user_agent_factory()
                controller = AgentController(
                    agent, training=self.config.user_agent_training, save_path=self.config.save_path
                )
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
            color = (0, 255, 0) if car.role == "GRACZ" else (255, 100, 100)
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
            "WASD/Strzałki – GRACZ",
            "R – Reset",
            "ESC – Wyjście",
            "Kropki na minimapie: pozycje aut",
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

    def isRunning(self, truncated, terminated) -> bool:
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
            dt = self.clock.tick(self.config.fps) / 1000.0

            if self.config.render:
                self.screen.fill((0, 0, 0))
                self.draw_headers()

            step_results = []
            if self.config.render:
                for car, vp in zip(self.cars, self.viewports):
                    obs, reward, terminated, truncated, _ = car.step(dt)
                    step_results.append((car, obs, reward, terminated, truncated))
                    if not self.isRunning(truncated, terminated):
                        running = False
                    frame = car.render_array()
                    vp.blit_env(self.screen, frame)
            else:
                for car in self.cars:
                    obs, reward, terminated, truncated, _ = car.step(dt)
                    step_results.append((car, obs, reward, terminated, truncated))
                    if not self.isRunning(truncated, terminated):
                        running = False
            if self.config.user_agent_training:
                for car, obs, reward, terminated, truncated in step_results:
                    if hasattr(car.controller, "observe"):
                        car.controller.observe(next_obs=obs, reward=reward, terminated=terminated, truncated=truncated)

            if self.config.render:
                self.draw_minimap()
                self.draw_instructions()

                pygame.display.flip()

                running = running and self.handle_events()

        for car in self.cars:
            car.close()

        if self.config.render:
            pygame.quit()
