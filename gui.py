import pygame
import numpy as np
import gymnasium as gym
import cv2


def create_full_track_minimap(env, size=200):
    env.reset()
    track_nodes = env.unwrapped.track

    if not track_nodes:
        return np.zeros((size, size, 3), dtype=np.uint8)

    x_coords = [node[2] for node in track_nodes]
    y_coords = [node[3] for node in track_nodes]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    margin = 50
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    width = max_x - min_x
    height = max_y - min_y
    scale = min(size / width, size / height) * 0.9

    offset_x = (size - width * scale) / 2
    offset_y = (size - height * scale) / 2

    minimap = np.zeros((size, size, 3), dtype=np.uint8)

    for i in range(len(track_nodes)):
        current = track_nodes[i]
        next_node = track_nodes[(i + 1) % len(track_nodes)]

        x1 = int((current[2] - min_x) * scale + offset_x)
        y1 = int((current[3] - min_y) * scale + offset_y)
        x2 = int((next_node[2] - min_x) * scale + offset_x)
        y2 = int((next_node[3] - min_y) * scale + offset_y)

        cv2.line(minimap, (x1, y1), (x2, y2), (255, 255, 255), 3)

    return minimap


def get_car_position_on_minimap(env, minimap_size=200):
    try:
        car = env.unwrapped.car
        if car is None:
            return minimap_size // 2, minimap_size // 2

        car_x = car.hull.position.x
        car_y = car.hull.position.y

        track_nodes = env.unwrapped.track
        if not track_nodes:
            return minimap_size // 2, minimap_size // 2

        x_coords = [node[2] for node in track_nodes]
        y_coords = [node[3] for node in track_nodes]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        margin = 50
        min_x -= margin
        max_x += margin
        min_y -= margin
        max_y += margin

        width = max_x - min_x
        height = max_y - min_y
        scale = min(minimap_size / width, minimap_size / height) * 0.9

        offset_x = (minimap_size - width * scale) / 2
        offset_y = (minimap_size - height * scale) / 2

        map_x = int((car_x - min_x) * scale + offset_x)
        map_y = int((car_y - min_y) * scale + offset_y)

        map_x = max(0, min(minimap_size - 1, map_x))
        map_y = max(0, min(minimap_size - 1, map_y))

        return map_x, map_y

    except Exception:
        return minimap_size // 2, minimap_size // 2


def play():
    env1 = gym.make("CarRacing-v3", render_mode="rgb_array", max_episode_steps=10000)
    env2 = gym.make("CarRacing-v3", render_mode="rgb_array", max_episode_steps=10000)
    FIXED_SEED = np.random.randint(0, 10000)

    obs1, info1 = env1.reset(seed=FIXED_SEED)
    obs2, info2 = env2.reset(seed=FIXED_SEED)

    track_map = create_full_track_minimap(env1)

    pygame.init()
    pygame.font.init()

    width, height = 1400, 900
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("CarRacing - Dual View")
    clock = pygame.time.Clock()

    minimap_size = 180
    track_surface = pygame.surfarray.make_surface(np.transpose(track_map, (1, 0, 2)))

    running = True
    while running:
        action1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        action2 = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action1[0] = -1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action1[0] = 1.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action1[1] = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action1[2] = 0.8

        obs1, reward1, terminated1, truncated1, info1 = env1.step(action1)
        obs2, reward2, terminated2, truncated2, info2 = env2.step(action2)

        screen.fill((0, 0, 0))

        frame1 = env1.render()
        surf1 = pygame.surfarray.make_surface(np.transpose(frame1, (1, 0, 2)))
        view_width = (width - minimap_size - 30) // 2
        surf1 = pygame.transform.smoothscale(surf1, (view_width, height - 100))
        screen.blit(surf1, (10, 50))

        frame2 = env2.render()
        surf2 = pygame.surfarray.make_surface(np.transpose(frame2, (1, 0, 2)))
        surf2 = pygame.transform.smoothscale(surf2, (view_width, height - 100))
        screen.blit(surf2, (20 + view_width, 50))

        pygame.draw.line(screen, (100, 100, 100), (10 + view_width, 0), (10 + view_width, height), 2)

        font_title = pygame.font.Font(None, 36)

        car1_title = font_title.render("GRACZ", True, (0, 255, 0))
        car1_rect = car1_title.get_rect(center=(view_width // 2, 25))
        pygame.draw.rect(screen, (0, 0, 0), car1_rect.inflate(20, 10))
        screen.blit(car1_title, car1_rect)

        car2_title = font_title.render("AI", True, (255, 100, 100))
        car2_rect = car2_title.get_rect(center=(20 + view_width + view_width // 2, 25))
        pygame.draw.rect(screen, (0, 0, 0), car2_rect.inflate(20, 10))
        screen.blit(car2_title, car2_rect)

        minimap_rect = pygame.Rect(width - minimap_size - 10, 10, minimap_size, minimap_size)
        pygame.draw.rect(screen, (0, 0, 0), minimap_rect)
        screen.blit(track_surface, (width - minimap_size - 10, 10))

        car1_x, car1_y = get_car_position_on_minimap(env1, minimap_size)
        car2_x, car2_y = get_car_position_on_minimap(env2, minimap_size)

        car1_screen_x = width - minimap_size - 10 + car1_x
        car1_screen_y = 10 + car1_y
        car2_screen_x = width - minimap_size - 10 + car2_x
        car2_screen_y = 10 + car2_y

        pygame.draw.circle(screen, (0, 255, 0), (car1_screen_x, car1_screen_y), 5)
        pygame.draw.circle(screen, (255, 100, 100), (car2_screen_x, car2_screen_y), 5)

        pygame.draw.rect(screen, (255, 255, 255), minimap_rect, 2)

        instructions = [
            "WASD/Strzałki - Samochód 1",
            "R - Reset",
            "ESC - Wyjście",
            "Zielony/Czerwony - pozycje aut"
        ]
        font_small = pygame.font.Font(None, 22)
        for i, instruction in enumerate(instructions):
            text = font_small.render(instruction, True, (200, 200, 200))
            text_rect = text.get_rect(bottomright=(width - 10, height - 10 - i * 25))
            screen.blit(text, text_rect)

        pygame.display.flip()

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_r:
                    obs1, info1 = env1.reset(seed=FIXED_SEED)
                    obs2, info2 = env2.reset(seed=FIXED_SEED)
                    track_map = create_full_track_minimap(env1)
                    track_surface = pygame.surfarray.make_surface(np.transpose(track_map, (1, 0, 2)))
                elif e.key == pygame.K_ESCAPE:
                    running = False

        clock.tick(60)

    env1.close()
    env2.close()
    pygame.quit()


if __name__ == "__main__":
    play()
