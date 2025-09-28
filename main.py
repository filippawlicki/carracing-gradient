import argparse
from multi_car_racing.Game import Game
from multi_car_racing.GameConfig import GameConfig
from multi_car_racing.controllers.PPOController import PPOController
from multi_car_racing.controllers.SB3Controller import SB3Controller

def make_ppo_agent(training=False):
    custom_config = {
        "lr": 3e-5,          # Lower learning rate
        "gamma": 0.99,       # Higher discount factor
        "eps_clip": 0.1,     # Tighter clipping
        "k_epochs": 4,       # Fewer update epochs
        "max_episodes": 50
    }
    agent = PPOController(training=training,
                     save_path='trained/ppo_last.pt', train_config=custom_config)
    if not training: 
        agent.load('trained/ppo_last.pt')
    return agent

def make_sb3_agent(training=False):
    agent = SB3Controller(training=training, save_path='trained/sb3_ppo.zip')
    if not training: 
        agent.load('trained/best_model.zip')
    return agent

def run_train():
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
    config = GameConfig(
        number_of_cars=2,
        user_agent_factory=lambda: make_sb3_agent(training=False),  
        user_agent_training=False,
        render=True,
        human=True,
        save_path='trained/sb3_ppo.zip'
    )
    Game(config).run()

def run_sb3_train():
    """Train using Stable Baselines3 PPO as baseline."""
    from train_sb3 import train_sb3_baseline
    train_sb3_baseline()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play_vs_ai", "train_sb3"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "play_vs_ai":
        run_play_vs_ai()
    elif args.mode == "train_sb3":
        run_sb3_train()
