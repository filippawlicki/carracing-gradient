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

        self.seed = self.config.seed if self.config.seed is not None else int(np.random.randint(0, 10_000))

        self.seed = random.choice([495, 1678, 2836])

        self.clock = pygame.time.Clock()
        self.screen: Optional[pygame.Surface] = None

        self.cars = []

        self.track_map = TrackMap(seed=self.seed, size=config.minimap_size)

        self.minimap = MiniMap(track=self.track_map)

        self.viewports: list[ViewportRenderer] = []

        self.step_idx = 0
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
        return gym.make("CarRacing-v3", render_mode=render_mode)

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
        assert self.screen and self.font_small

        pad = 3
        title_h = 22
        radius = 10
        margin = 1

        panel_w = self.config.minimap_size + 2 * pad
        panel_h = self.config.minimap_size + 2 * pad + title_h + 1

        panel_x = self.config.width - panel_w - margin
        panel_y = margin
        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)

        shadow = panel_rect.copy()
        shadow.x += 2
        shadow.y += 2
        pygame.draw.rect(self.screen, (0, 0, 0), shadow, border_radius=radius)

        pygame.draw.rect(self.screen, (20, 20, 20), panel_rect, border_radius=radius)
        pygame.draw.rect(self.screen, (70, 70, 70), panel_rect, width=2, border_radius=radius)

        title_rect = pygame.Rect(panel_rect.x, panel_rect.y, panel_rect.w, title_h)
        pygame.draw.rect(self.screen, (30, 30, 30), title_rect, border_radius=radius)
        pygame.draw.rect(self.screen, (30, 30, 30),
                         pygame.Rect(title_rect.x, title_rect.y + radius, title_rect.w, title_rect.h - radius))
        title_text = self.font_small.render("Minimap", True, (200, 200, 200))
        self.screen.blit(title_text, title_text.get_rect(midleft=(title_rect.x + pad + 2, title_rect.centery)))

        map_rect = pygame.Rect(
            panel_rect.x + pad,
            panel_rect.y + pad + title_h,
            self.config.minimap_size,
            self.config.minimap_size
        )

        surf = self.minimap.to_surface()
        pygame.draw.rect(self.screen, (0, 0, 0), map_rect, border_radius=6)
        self.screen.blit(surf, map_rect.topleft)

        players = []
        for car in self.cars:
            assert car
            car_x, car_y = car.car_xy()
            players.append(self.track_map.world_to_minimap(car_x, car_y))

        self.minimap.draw_cars(self.screen, map_rect.topleft, players)

    def draw_stopper(self) -> None:
        assert self.screen and self.font_small

        total_ms = int(self.step_idx)
        s = total_ms // 100
        ms = total_ms % 100
        time_text = f"Time: {s}:{ms}"

        minimap_x = self.config.width - self.config.minimap_size + 180
        minimap_y = 220

        panel_w = 160
        panel_h = 40
        panel_rect = pygame.Rect(minimap_x - panel_w - 10, minimap_y, panel_w, panel_h)

        pygame.draw.rect(self.screen, (20, 20, 20), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (70, 70, 70), panel_rect, width=2, border_radius=10)

        text = self.font_small.render(time_text, True, (230, 230, 230))
        text_rect = text.get_rect(center=panel_rect.center)
        self.screen.blit(text, text_rect)

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

    def is_running(self, truncated, terminated) -> bool:
        return not (terminated or truncated)

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

    def run(self) -> None:
        if self.config.render:
            self.init_pygame()
            assert self.screen
        self.init_envs()

        for car in self.cars:
            assert car

        end = False
        running = True
        while running:
            if self.config.render:
                dt = self.clock.tick(self.config.fps) / 1000.0
            else:
                dt = 0

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
                    end = not self.is_running(truncated, terminated)
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
                self.draw_stopper()
                self.draw_instructions()
                pygame.display.flip()
                running = running and self.handle_events()

            self.step_idx += 1

            if not self.config.user_agent_training and end:
                for car in self.cars:
                    car.reset()
                    end = False
                    self.step_idx = 0

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