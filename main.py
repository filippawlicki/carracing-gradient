import argparse
import os
from multi_car_racing.Game import Game
from multi_car_racing.GameConfig import GameConfig
from multi_car_racing.controllers.PPOController import PPOController
from multi_car_racing.controllers.SB3Controller import SB3Controller


def make_ppo_agent(training=False):
    custom_config = {
        "lr": 3e-4,
        "gamma": 0.99,
        "eps_clip": 0.2,
        "k_epochs": 4,
        "max_episodes": 100,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
    }
    agent = PPOController(
        training=training,
        save_path='trained/ppo_last.pt',
        train_config=custom_config
    )
    if not training:
        agent.load('trained/ppo_last.pt')
    return agent


def make_sb3_agent(training=False, model_path='trained/sb3_ppo.zip'):
    agent = SB3Controller(
        training=training,
        save_path='trained/sb3_ppo.zip'
    )
    if not training:
        try:
            agent.load(model_path)
        except FileNotFoundError:
            print(f"Warning: Could not find {model_path}")
            raise
    return agent


def run_train():
    print("[Main] Starting custom PPO training...")
    config = GameConfig(
        number_of_cars=1,
        user_agent_factory=lambda: make_ppo_agent(training=True),
        user_agent_training=True,
        render=False,
        human=False,
        save_path='trained/ppo_last.pt'
    )
    Game(config).run()


def run_play_vs_ai():
    print("Starting Player vs AI mode...")

    # Check what models are available
    models = {
        'sb3': 'trained/sb3_ppo.zip',
        'custom': 'trained/ppo_last.pt'
    }

    available = {}
    for name, path in models.items():
        if os.path.exists(path):
            available[name] = path
            print(f"Found {name} model: {path}")

    if not available:
        print("\nNo trained models found!")
        print("Please train a model first:")
        print("  - python main.py --mode train_sb3")
        print("  - python main.py --mode train")
        return

    # Prefer SB3 if available, otherwise use custom
    if 'sb3' in available:
        model_path = available['sb3']
        agent_factory = lambda: make_sb3_agent(training=False, model_path=model_path)
        print(f"Using SB3 agent")
    else:
        model_path = available['custom']
        agent_factory = lambda: make_ppo_agent(training=False)
        print(f"Using custom PPO agent")

    config = GameConfig(
        number_of_cars=2,
        user_agent_factory=agent_factory,
        user_agent_training=False,
        render=True,
        human=True,
        save_path=model_path
    )
    Game(config).start_game()


def run_sb3_train():
    print("Starting SB3 PPO training...")
    from train_sb3 import train_sb3_baseline
    train_sb3_baseline()


def run_watch_custom():
    print("Watching custom PPO agent...")
    config = GameConfig(
        number_of_cars=1,
        user_agent_factory=lambda: make_ppo_agent(training=False),
        user_agent_training=False,
        render=True,
        human=False,
    )
    Game(config).run()


def run_ai_vs_ai():
    print("Starting AI vs AI mode...")

    def agent_factory():
        if not hasattr(agent_factory, 'counter'):
            agent_factory.counter = 0

        if agent_factory.counter == 0:
            agent = make_sb3_agent(training=False)
        else:
            agent = make_ppo_agent(training=False)

        agent_factory.counter += 1
        return agent

    config = GameConfig(
        number_of_cars=2,
        user_agent_factory=agent_factory,
        user_agent_training=False,
        render=True,
        human=False,
    )
    Game(config).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Car Racing Multi-Agent System")
    parser.add_argument(
        "--mode",
        choices=["train", "play_vs_ai", "train_sb3", "watch_custom", "ai_vs_ai"],
        default="play_vs_ai",
        help="Mode to run: train (custom PPO), play_vs_ai (human vs SB3), "
             "train_sb3 (train SB3 baseline), watch_custom (watch your agent), "
             "ai_vs_ai (SB3 vs custom PPO)"
    )
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "play_vs_ai":
        run_play_vs_ai()
    elif args.mode == "train_sb3":
        run_sb3_train()
    elif args.mode == "watch_custom":
        run_watch_custom()
    elif args.mode == "ai_vs_ai":
        run_ai_vs_ai()