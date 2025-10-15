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
        self.winner_banner: Optional[str] = None

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
        """
        Tworzy środowisko CarRacing-v3 bez HUD (pasków na dole).
        """
        render_mode = None if not self.config.render else "rgb_array"
        env = gym.make("CarRacing-v3", render_mode=render_mode)

        # Wyłącz HUD w każdym środowisku
        if hasattr(env.unwrapped, "render"):
            env.unwrapped.render_hud = False  # <- kluczowe, żeby nie było pasków

        return env

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
                pygame.quit()
                exit()
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
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

    def show_end_screen(self):
        if not self.config.render:
            return

        assert self.screen and self.font_title

        clock = pygame.time.Clock()
        t = 0
        waiting = True

        # Tekst zależny od zwycięzcy
        if self.winner == "PLAYER":
            winner_text = "WYGRAŁ GRACZ!"
            color = (0, 255, 0)
        else:
            winner_text = "WYGRAŁO AI!"
            color = (255, 100, 100)

        button_font = pygame.font.Font(None, int(self.config.height * 0.07))
        menu_rect = pygame.Rect(0, 0, 300, 100)
        menu_rect.center = (self.config.width // 2, self.config.height * 2 // 3)

        while waiting:
            self.screen.fill((10, 10, 10))

            # 🔹 Tytuł – pulsujący
            pulse = (np.sin(t * 0.1) + 1) / 2
            scale = 1.0 + 0.05 * pulse
            font_size = int(self.config.height * 0.12 * scale)
            font = pygame.font.Font(None, font_size)
            title_surface = font.render(winner_text, True, color)
            title_rect = title_surface.get_rect(center=(self.config.width // 2, self.config.height // 3))
            self.screen.blit(title_surface, title_rect)

            # 🔹 Przycisk MENU
            mouse_pos = pygame.mouse.get_pos()
            hover = menu_rect.collidepoint(mouse_pos)
            menu_color = tuple(min(255, c + 40) for c in (100, 200, 255)) if hover else (100, 200, 255)
            pygame.draw.rect(self.screen, menu_color, menu_rect, border_radius=25)
            pygame.draw.rect(self.screen, (50, 50, 50), menu_rect, width=4, border_radius=25)

            text_surf = button_font.render("MENU", True, (0, 0, 0))
            self.screen.blit(text_surf, text_surf.get_rect(center=menu_rect.center))

            pygame.display.flip()
            t += 1
            clock.tick(60)

            # Obsługa zdarzeń
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        exit()
                    elif event.key == pygame.K_RETURN:
                        waiting = False  # powrót do menu
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if menu_rect.collidepoint(event.pos):
                        waiting = False  # powrót do menu

    def show_start_screen(self):
        if not self.config.render:
            return

        assert self.screen and self.font_title

        # Czcionki
        title_font = pygame.font.Font(None, int(self.config.height * 0.18))
        button_font = pygame.font.Font(None, int(self.config.height * 0.07))
        info_font = pygame.font.Font(None, 40)

        title_text = " CAR RACING "

        # Przyciski START i EXIT
        button_w, button_h = 300, 100
        start_rect = pygame.Rect(0, 0, button_w, button_h)
        exit_rect = pygame.Rect(0, 0, button_w, button_h)
        start_rect.center = (self.config.width // 2, self.config.height * 2 // 3 - 80)
        exit_rect.center = (self.config.width // 2, self.config.height * 2 // 3 + 80)

        clock = pygame.time.Clock()
        t = 0
        waiting = True

        while waiting:
            self.screen.fill((10, 10, 10))

            # 🔹 Tło – asfalt z liniami wyścigowymi
            # Linie boczne toru
            track_width = 300
            center_x = self.config.width // 2
            pygame.draw.rect(self.screen, (40, 40, 40),
                             (center_x - track_width // 2, 0, track_width, self.config.height))

            # Białe linie boczne
            pygame.draw.line(self.screen, (200, 200, 200),
                             (center_x - track_width // 2, 0),
                             (center_x - track_width // 2, self.config.height), 5)
            pygame.draw.line(self.screen, (200, 200, 200),
                             (center_x + track_width // 2, 0),
                             (center_x + track_width // 2, self.config.height), 5)

            # Przerywana linia środkowa (animowana)
            dash_length = 30
            gap_length = 20
            y_offset = (t * 8) % (dash_length + gap_length)
            for i in range(-1, self.config.height // (dash_length + gap_length) + 2):
                y_start = i * (dash_length + gap_length) - y_offset
                pygame.draw.line(self.screen, (255, 255, 100),
                                 (center_x, y_start),
                                 (center_x, y_start + dash_length), 6)

            # 🔹 Tytuł – pulsujący z efektem
            pulse = (np.sin(t * 0.1) + 1) / 2
            scale = 1.0 + 0.08 * pulse
            font_size = int(self.config.height * 0.15 * scale)
            font = pygame.font.Font(None, font_size)
            title_surface = font.render(title_text, True, (255, 200 + int(55 * pulse), 50))
            title_rect = title_surface.get_rect(center=(self.config.width // 2, self.config.height // 4))

            # Cień tytułu
            shadow = font.render(title_text, True, (0, 0, 0))
            self.screen.blit(shadow, (title_rect.x + 4, title_rect.y + 4))
            self.screen.blit(title_surface, title_rect)

            # 🔹 Rysowanie przycisków START i EXIT
            mouse_pos = pygame.mouse.get_pos()
            for label, rect, base_color in [
                ("START", start_rect, (100, 255, 100)),
                ("WYJŚCIE", exit_rect, (255, 100, 100))
            ]:
                hover = rect.collidepoint(mouse_pos)
                color = tuple(min(255, c + 40) for c in base_color) if hover else base_color

                # Cień przycisku
                shadow_rect = rect.copy()
                shadow_rect.x += 4
                shadow_rect.y += 4
                pygame.draw.rect(self.screen, (0, 0, 0), shadow_rect, border_radius=25)

                pygame.draw.rect(self.screen, color, rect, border_radius=25)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, width=4, border_radius=25)

                text_surf = button_font.render(label, True, (0, 0, 0))
                self.screen.blit(text_surf, text_surf.get_rect(center=rect.center))

            # 🔹 Samochody wyścigowe z perspektywą 3D
            bounce = int(5 * np.sin(t * 0.15))  # Animacja odbijania

            # Lewy samochód (gracz - zielony wyścigówka)
            car_x_left = start_rect.centerx - 280
            car_y_left = start_rect.centery - 50 + bounce

            # Cień samochodu
            shadow_points = [(car_x_left + 20, car_y_left + 140), (car_x_left + 90, car_y_left + 140),
                             (car_x_left + 80, car_y_left + 150), (car_x_left + 30, car_y_left + 150)]
            pygame.draw.polygon(self.screen, (20, 20, 20, 100), shadow_points)

            # Tylna część (silnik)
            pygame.draw.rect(self.screen, (0, 150, 0), (car_x_left + 25, car_y_left + 90, 60, 40), border_radius=8)

            # Główny korpus (aerodynamiczny kształt)
            body_points = [
                (car_x_left + 55, car_y_left),  # Przód (nos)
                (car_x_left + 20, car_y_left + 40),  # Lewy bok
                (car_x_left + 20, car_y_left + 100),  # Lewy tył
                (car_x_left + 90, car_y_left + 100),  # Prawy tył
                (car_x_left + 90, car_y_left + 40),  # Prawy bok
            ]
            pygame.draw.polygon(self.screen, (0, 220, 0), body_points)

            # Błysk na karoserii (efekt metaliczny)
            highlight_points = [
                (car_x_left + 55, car_y_left + 10),
                (car_x_left + 30, car_y_left + 45),
                (car_x_left + 35, car_y_left + 45),
                (car_x_left + 55, car_y_left + 15),
            ]
            pygame.draw.polygon(self.screen, (100, 255, 100), highlight_points)

            # Kabina/kokpit
            cockpit = [(car_x_left + 55, car_y_left + 30), (car_x_left + 40, car_y_left + 50),
                       (car_x_left + 70, car_y_left + 50)]
            pygame.draw.polygon(self.screen, (50, 150, 200), cockpit)
            pygame.draw.polygon(self.screen, (255, 255, 255), cockpit, 2)

            # Reflektory LED
            pygame.draw.circle(self.screen, (255, 255, 255), (car_x_left + 45, car_y_left + 8), 6)
            pygame.draw.circle(self.screen, (255, 255, 255), (car_x_left + 65, car_y_left + 8), 6)
            pygame.draw.circle(self.screen, (255, 255, 100), (car_x_left + 45, car_y_left + 8), 4)
            pygame.draw.circle(self.screen, (255, 255, 100), (car_x_left + 65, car_y_left + 8), 4)

            # Spoiler tylny
            pygame.draw.rect(self.screen, (0, 100, 0), (car_x_left + 15, car_y_left + 85, 80, 8), border_radius=2)
            pygame.draw.rect(self.screen, (0, 180, 0), (car_x_left + 15, car_y_left + 82, 80, 3))

            # Koła z bieżnikiem
            for wheel_y in [car_y_left + 35, car_y_left + 85]:
                for wheel_x in [car_x_left + 15, car_x_left + 95]:
                    pygame.draw.circle(self.screen, (20, 20, 20), (wheel_x, wheel_y), 16)
                    pygame.draw.circle(self.screen, (60, 60, 60), (wheel_x, wheel_y), 12)
                    pygame.draw.circle(self.screen, (100, 100, 100), (wheel_x, wheel_y), 8)
                    # Szprychy obracające się
                    angle = t * 0.3
                    for i in range(4):
                        a = angle + i * np.pi / 2
                        x1 = wheel_x + 4 * np.cos(a)
                        y1 = wheel_y + 4 * np.sin(a)
                        x2 = wheel_x + 10 * np.cos(a)
                        y2 = wheel_y + 10 * np.sin(a)
                        pygame.draw.line(self.screen, (150, 150, 150), (x1, y1), (x2, y2), 2)

            # Numer wyścigowy
            number_font = pygame.font.Font(None, 40)
            num_surf = number_font.render("1", True, (255, 255, 255))
            self.screen.blit(num_surf, (car_x_left + 48, car_y_left + 60))

            # Prawy samochód (AI - czerwona wyścigówka)
            car_x_right = start_rect.centerx + 200
            car_y_right = start_rect.centery - 50 - bounce

            # Cień samochodu
            shadow_points = [(car_x_right + 20, car_y_right + 140), (car_x_right + 90, car_y_right + 140),
                             (car_x_right + 80, car_y_right + 150), (car_x_right + 30, car_y_right + 150)]
            pygame.draw.polygon(self.screen, (20, 20, 20, 100), shadow_points)

            # Tylna część (silnik)
            pygame.draw.rect(self.screen, (150, 0, 0), (car_x_right + 25, car_y_right + 90, 60, 40), border_radius=8)

            # Główny korpus
            body_points = [
                (car_x_right + 55, car_y_right),
                (car_x_right + 20, car_y_right + 40),
                (car_x_right + 20, car_y_right + 100),
                (car_x_right + 90, car_y_right + 100),
                (car_x_right + 90, car_y_right + 40),
            ]
            pygame.draw.polygon(self.screen, (255, 40, 40), body_points)

            # Błysk metaliczny
            highlight_points = [
                (car_x_right + 55, car_y_right + 10),
                (car_x_right + 30, car_y_right + 45),
                (car_x_right + 35, car_y_right + 45),
                (car_x_right + 55, car_y_right + 15),
            ]
            pygame.draw.polygon(self.screen, (255, 150, 150), highlight_points)

            # Kabina/kokpit
            cockpit = [(car_x_right + 55, car_y_right + 30), (car_x_right + 40, car_y_right + 50),
                       (car_x_right + 70, car_y_right + 50)]
            pygame.draw.polygon(self.screen, (50, 150, 200), cockpit)
            pygame.draw.polygon(self.screen, (255, 255, 255), cockpit, 2)

            # Reflektory LED
            pygame.draw.circle(self.screen, (255, 255, 255), (car_x_right + 45, car_y_right + 8), 6)
            pygame.draw.circle(self.screen, (255, 255, 255), (car_x_right + 65, car_y_right + 8), 6)
            pygame.draw.circle(self.screen, (255, 255, 100), (car_x_right + 45, car_y_right + 8), 4)
            pygame.draw.circle(self.screen, (255, 255, 100), (car_x_right + 65, car_y_right + 8), 4)

            # Spoiler tylny
            pygame.draw.rect(self.screen, (100, 0, 0), (car_x_right + 15, car_y_right + 85, 80, 8), border_radius=2)
            pygame.draw.rect(self.screen, (255, 100, 100), (car_x_right + 15, car_y_right + 82, 80, 3))

            # Koła z bieżnikiem
            for wheel_y in [car_y_right + 35, car_y_right + 85]:
                for wheel_x in [car_x_right + 15, car_x_right + 95]:
                    pygame.draw.circle(self.screen, (20, 20, 20), (wheel_x, wheel_y), 16)
                    pygame.draw.circle(self.screen, (60, 60, 60), (wheel_x, wheel_y), 12)
                    pygame.draw.circle(self.screen, (100, 100, 100), (wheel_x, wheel_y), 8)
                    # Szprychy obracające się
                    angle = -t * 0.3
                    for i in range(4):
                        a = angle + i * np.pi / 2
                        x1 = wheel_x + 4 * np.cos(a)
                        y1 = wheel_y + 4 * np.sin(a)
                        x2 = wheel_x + 10 * np.cos(a)
                        y2 = wheel_y + 10 * np.sin(a)
                        pygame.draw.line(self.screen, (150, 150, 150), (x1, y1), (x2, y2), 2)

            # Numer wyścigowy
            num_surf = number_font.render("2", True, (255, 255, 255))
            self.screen.blit(num_surf, (car_x_right + 48, car_y_right + 60))

            # Napisy pod samochodami z efektami
            label_font = pygame.font.Font(None, 36)

            # Efekt świetlny pod napisem GRACZ
            glow_pulse = int(30 * (np.sin(t * 0.2) + 1) / 2)
            player_label = label_font.render("GRACZ", True, (100 + glow_pulse, 255, 100 + glow_pulse))
            player_shadow = label_font.render("GRACZ", True, (0, 50, 0))
            player_pos = (car_x_left + 55, car_y_left + 155)
            self.screen.blit(player_shadow, (player_pos[0] - 28, player_pos[1] + 2))
            self.screen.blit(player_label, (player_pos[0] - 30, player_pos[1]))

            # Efekt świetlny pod napisem AI
            ai_label = label_font.render("AI", True, (255, 100 + glow_pulse, 100 + glow_pulse))
            ai_shadow = label_font.render("AI", True, (50, 0, 0))
            ai_pos = (car_x_right + 55, car_y_right + 155)
            self.screen.blit(ai_shadow, (ai_pos[0] - 10, ai_pos[1] + 2))
            self.screen.blit(ai_label, (ai_pos[0] - 12, ai_pos[1]))

            # 🔹 Info na dole
            info_surf = info_font.render("Naciśnij ENTER lub kliknij START", True, (220, 220, 220))
            self.screen.blit(info_surf, info_surf.get_rect(center=(self.config.width // 2, self.config.height - 50)))

            pygame.display.flip()
            t += 1
            clock.tick(60)

            # Obsługa zdarzeń
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        exit()
                    elif event.key == pygame.K_RETURN:
                        waiting = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if start_rect.collidepoint(event.pos):
                        waiting = False
                    elif exit_rect.collidepoint(event.pos):
                        pygame.quit()
                        exit()

    def countdown(self):
        if not self.config.render:
            return

        assert self.screen and self.font_title

        big_font_size = int(self.config.height * 0.4)
        big_font = pygame.font.Font(None, big_font_size)
        clock = pygame.time.Clock()

        sequence = [
            ("3", (255, 0, 0)),
            ("2", (255, 140, 0)),
            ("1", (255, 255, 0)),
            ("START!", (0, 255, 0)),
        ]

        for num, color in sequence:
            t = 0
            duration = 60  # ok. 1 sekunda (60 klatek)
            while t < duration:
                self.screen.fill((0, 0, 0))
                pulse = (np.sin(t * 0.2) + 1) / 2  # efekt „pulsowania"
                scale = 1.0 + 0.15 * pulse
                font_size = int(big_font_size * scale)
                font = pygame.font.Font(None, font_size)

                # render tekstu
                text = font.render(num, True, color)
                rect = text.get_rect(center=(self.config.width // 2, self.config.height // 2))

                # cień dla lepszego efektu
                shadow = font.render(num, True, (0, 0, 0))
                self.screen.blit(shadow, (rect.x + 5, rect.y + 5))
                self.screen.blit(text, rect)

                pygame.display.flip()
                clock.tick(60)
                t += 1

                # Obsługa ESC podczas odliczania
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            exit()

        self.clock.tick()

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
                self.winner_banner = self.winner

                self.winner = None
                end = False
                running = False
                self.step_idx = 0

            if self.config.render:
                self.draw_minimap()
                self.draw_stopper()
              #  self.draw_instructions()
                # 🔹 Stałe napisy GRACZ i AI na dole ekranu
                assert self.font_small
                label_font = self.font_small

                # Pozycje napisów (na dole ekranu, trochę nad dolną krawędzią)
                screen_center_x = self.config.width // 2
                bottom_y = self.config.height - 30

                # GRACZ – zielony
                player_label = label_font.render("GRACZ", True, (0, 255, 0))
                player_rect = player_label.get_rect(center=(screen_center_x - 100, bottom_y))
                self.screen.blit(player_label, player_rect)

                # AI – czerwony
                ai_label = label_font.render("AI", True, (255, 0, 0))
                ai_rect = ai_label.get_rect(center=(screen_center_x + 100, bottom_y))
                self.screen.blit(ai_label, ai_rect)

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

    def start_game(self) -> None:
        """Główna pętla aplikacji – menu → gra → ekran końcowy → powrót do menu."""
        if self.config.render:
            self.init_pygame()

        app_running = True
        while app_running:
            # 🔹 EKRAN STARTOWY
            if self.config.render:
                self.show_start_screen()

            # 🔹 Przygotowanie środowiska gry
            self.cars.clear()
            self.track_map = TrackMap(seed=self.seed, size=self.config.minimap_size)
            self.minimap = MiniMap(track=self.track_map)
            self.viewports.clear()
            self.winner = None
            self.winner_banner = None

            self.init_envs()

            # 🔹 Odliczanie
            if self.config.render:
                self.countdown()

            # 🔹 Uruchomienie rozgrywki
            self.run()

            # 🔹 EKRAN KOŃCOWY
            if self.config.render:
                self.show_end_screen()

            # 🔹 Po zakończeniu pytamy gracza, czy chce zagrać ponownie
            # (Ekran końcowy już ma przycisk MENU, który prowadzi do show_start_screen)
            # Tu tylko sprawdzamy, czy użytkownik nie zamknął gry
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        exit()

            # 🔹 Jeśli użytkownik kliknie "MENU" w show_end_screen(),
            # pętla wróci automatycznie do początku (czyli do show_start_screen)

        # 🔹 Zakończenie aplikacji
        if self.config.render:
            pygame.quit()
