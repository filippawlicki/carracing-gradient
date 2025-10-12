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

        self.seed = random.choice([495, 1678, 2836])

        self.clock = pygame.time.Clock()
        self.screen: Optional[pygame.Surface] = None

        self.cars: list[EnvCar] = []

        self.track_map = TrackMap(seed=self.seed, size=config.minimap_size)
        self.minimap = MiniMap(track=self.track_map)

        self.viewports: list[ViewportRenderer] = []

        self.finish_line: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        self.winner: Optional[str] = None

        self.step_idx: int = 0

        self.font_title: Optional[pygame.font.Font] = None
        self.font_small: Optional[pygame.font.Font] = None

    def init_finish_line(self, env) -> None:

        nodes = env.unwrapped.track
        assert nodes and len(nodes) >= 2

        x0, y0 = nodes[0][2], nodes[0][3]
        x1, y1 = nodes[1][2], nodes[1][3]

        dx, dy = (x1 - x0), (y1 - y0)
        nx, ny = -dy, dx
        norm = (nx ** 2 + ny ** 2) ** 0.5 or 1.0
        nx, ny = nx / norm, ny / norm

        half_len = 20.0
        p1 = (x0 - nx * half_len, y0 - ny * half_len)
        p2 = (x0 + nx * half_len, y0 + ny * half_len)
        self.finish_line = (p1, p2)

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
        self.init_finish_line(base_env)

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

            cx, cy = car.car_xy()
            car.prev_pos = (cx, cy)
            car.progress = 0.0

            self.cars.append(car)

    def reset_both(self) -> None:
        for car in self.cars:
            car.reset(self.seed)
            if car is self.cars[-1]:
                self.track_map.build_from_env(car.env)
        self.step_idx = 0
        self.winner = None
        for car in self.cars:
            cx, cy = car.car_xy()
            car.prev_pos = (cx, cy)
            car.progress = 0.0

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
        pygame.draw.rect(
            self.screen, (30, 30, 30),
            pygame.Rect(title_rect.x, title_rect.y + radius, title_rect.w, title_rect.h - radius)
        )
        title_text = self.font_small.render("Minimap", True, (200, 200, 200))
        self.screen.blit(title_text, title_text.get_rect(midleft=(title_rect.x + pad + 2, title_rect.centery)))

        map_rect = pygame.Rect(
            panel_rect.x + pad,
            panel_rect.y + pad + title_h,
            self.config.minimap_size,
            self.config.minimap_size
        )

        pygame.draw.rect(self.screen, (0, 0, 0), map_rect, border_radius=6)

        surf = self.minimap.to_surface()
        self.screen.blit(surf, map_rect.topleft)

        players = []
        for car in self.cars:
            car_x, car_y = car.car_xy()
            players.append(self.track_map.world_to_minimap(car_x, car_y))
        self.minimap.draw_cars(self.screen, map_rect.topleft, players)

        if self.finish_line is not None:
            (x1, y1), (x2, y2) = self.finish_line
            m1 = self.track_map.world_to_minimap(x1, y1)
            m2 = self.track_map.world_to_minimap(x2, y2)
            base = map_rect.topleft
            pygame.draw.line(
                self.screen, (255, 255, 255),
                (base[0] + m1[0], base[1] + m1[1]),
                (base[0] + m2[0], base[1] + m2[1]), 3
            )

    def draw_stopper(self) -> None:
        assert self.screen and self.font_small

        total_ms = int(self.step_idx)
        s = total_ms // 1000
        ms = total_ms % 1000
        time_text = f"Time: {s}:{ms}"

        pad = 8
        panel_w, panel_h = 160, 40
        minimap_panel_w = self.config.minimap_size + 2 * 3
        margin = 10
        panel_x = self.config.width - minimap_panel_w - margin - 10 - panel_w + 190
        panel_y = margin + 200

        panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
        pygame.draw.rect(self.screen, (20, 20, 20), panel_rect, border_radius=10)
        pygame.draw.rect(self.screen, (70, 70, 70), panel_rect, width=2, border_radius=10)

        text = self.font_small.render(time_text, True, (230, 230, 230))
        self.screen.blit(text, text.get_rect(center=panel_rect.center))

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

    def segments_intersect(self, p, p2, q, q2) -> bool:
        def orient(a, b, c):
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        o1 = orient(p, p2, q)
        o2 = orient(p, p2, q2)
        o3 = orient(q, q2, p)
        o4 = orient(q, q2, p2)
        return (o1 == 0 and o2 == 0 and o3 == 0 and o4 == 0) or (o1 * o2 < 0 and o3 * o4 < 0)

    def _update_progress(self, env, car, x, y):
        nodes = env.unwrapped.track
        if not nodes:
            return
        best_idx, best_d2 = 0, float("inf")
        for i, n in enumerate(nodes):
            nx, ny = n[2], n[3]
            d2 = (nx - x) ** 2 + (ny - y) ** 2
            if d2 < best_d2:
                best_d2, best_idx = d2, i
        total = len(nodes)
        prev = int(getattr(car, "progress", 0.0) * total + 0.5)
        if best_idx > prev:
            car.progress = best_idx / total

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
                dt = 0.0

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

                    if self.finish_line is not None:
                        (fx1, fy1), (fx2, fy2) = self.finish_line
                        cx, cy = car.car_xy()
                        px, py = getattr(car, "prev_pos", (cx, cy))

                        crossed = self.segments_intersect((px, py), (cx, cy), (fx1, fy1), (fx2, fy2))

                        self._update_progress(car.env, car, cx, cy)

                        if crossed and getattr(car, "progress", 0.0) >= 0.6 and self.winner is None:
                            self.winner = car.role
                            end = True

                        car.prev_pos = (cx, cy)
            else:
                for car in self.cars:
                    obs, reward, terminated, truncated, _ = car.step(dt)
                    step_results.append((car, obs, reward, terminated, truncated))

            if self.config.user_agent_training:
                for car, obs, reward, terminated, truncated in step_results:
                    if hasattr(car.controller, "observe"):
                        car.controller.observe(next_obs=obs, reward=reward, terminated=terminated, truncated=truncated)
                        if getattr(car.controller, "_episode_done", False):
                            car.reset(self.seed)

            if not self.config.user_agent_training and end:
                for car in self.cars:
                    car.reset(self.seed)
                    cx, cy = car.car_xy()
                    car.prev_pos = (cx, cy)
                    car.progress = 0.0
                self.winner = None
                end = False
                self.step_idx = 0

            if self.config.render:
                self.draw_minimap()
                self.draw_stopper()
                self.draw_instructions()

                if self.winner and self.font_title:
                    msg = f"FINISH! Winner: {self.winner}"
                    banner = self.font_title.render(msg, True, (255, 255, 0))
                    rect = banner.get_rect(center=(self.config.width // 2, 30))
                    pygame.draw.rect(self.screen, (0, 0, 0), rect.inflate(20, 10))
                    self.screen.blit(banner, rect)

                pygame.display.flip()
                running = running and self.handle_events()

            self.step_idx += int(dt * 1000)

            if self.config.user_agent_training:
                all_done = all(getattr(car.controller, "training_done", lambda: False)() for car in self.cars)
                if all_done:
                    print("[Game] All agents finished training. Exiting.")
                    break

        for car in self.cars:
            car.close()

        if self.config.render:
            pygame.quit()
