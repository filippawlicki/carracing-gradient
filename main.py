import argparse
from multi_car_racing.Game import Game
from multi_car_racing.GameConfig import GameConfig
from multi_car_racing.controllers.PPOController import PPOController

def make_ppo_agent(training=False):
    agent = PPOController(training=training,
                     save_path='trained/ppo_last.pt')
    if not training: # If we are not training, load the trained model
        agent.load('trained/ppo_last.pt')
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
        user_agent_factory=lambda: make_ppo_agent(training=False),
        user_agent_training=False,
        render=True,
        human=True,
        save_path='trained/ppo_last.pt'
    )
    Game(config).run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "play_vs_ai"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "play_vs_ai":
        run_play_vs_ai()
