import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import pygame
from pygame.locals import *
import torch

MODEL_PATH = "ppo_carracing"

class HumanController:
    def __init__(self):
        pygame.init()
        self.action = np.array([0.0, 0.0, 0.0])  # [steer, gas, brake]
        self.done = False

    def get_action(self):
        keys = pygame.key.get_pressed()
        action = np.array([0.0, 0.0, 0.0])

        if keys[K_LEFT] or keys[K_a]:
            action[0] = -1.0
        if keys[K_RIGHT] or keys[K_d]:
            action[0] = 1.0
        if keys[K_UP] or keys[K_w]:
            action[1] = 1.0
        if keys[K_DOWN] or keys[K_s]:
            action[2] = 0.8  # brake

        for event in pygame.event.get():
            if event.type == QUIT:
                self.done = True
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                self.done = True

        return action

def train():
    env = make_vec_env("CarRacing-v3", n_envs=1)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO("CnnPolicy", env, verbose=1, device=device)
    model.learn(total_timesteps=100_000)
    model.save(MODEL_PATH)
    env.close()
    print("Model saved.")

def play_against_ai():
    env = gym.make("CarRacing-v3", render_mode="human")
    model = PPO.load(MODEL_PATH)
    human = HumanController()

    obs, _ = env.reset()
    clock = pygame.time.Clock()
    total_reward = 0

    while not human.done:
        action_human = human.get_action()

        action_ai, _ = model.predict(obs)

        use_ai = True # Set to True to use AI, False to use human input
        action = action_ai if use_ai else action_human

        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        clock.tick(60)

        if terminated or truncated:
            print("Episode finished. Total reward:", round(total_reward, 2))
            obs, _ = env.reset()
            total_reward = 0

    env.close()
    pygame.quit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--play", action="store_true", help="Play against AI")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.play:
        play_against_ai()
    else:
        print("Use --train or --play")

if __name__ == "__main__":
    main()
