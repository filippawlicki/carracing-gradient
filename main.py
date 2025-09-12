import argparse
from multi_car_racing.Game import Game
from multi_car_racing.GameConfig import GameConfig
from multi_car_racing.agents.simple_agent import SimpleAgent
from multi_car_racing.agents.ppo_agent import PPOAgent

def make_simple_agent():
    return SimpleAgent()

def make_ppo_agent():
    agent = PPOAgent()
    #agent.load('trained/ppo_last.pt')
    return agent

def run_train():
    # run PPO training loop
    config = GameConfig(
        number_of_cars=1,
        user_agent_factory=make_ppo_agent,
        user_agent_training=True,
        render=False,
        human=False,
        save_path='trained/ppo_last.pt'
    )
    Game(config).run()

def run_play_vs_simple():
    config = GameConfig(
        number_of_cars=2,
        user_agent_factory=make_simple_agent,
        user_agent_training=False,
        render=True,
        human=True,
        save_path='trained/ppo_last.pt'
    )
    Game(config).run()

def run_play_vs_ai():
    config = GameConfig(
        number_of_cars=2,
        user_agent_factory=make_ppo_agent,
        user_agent_training=False,
        render=True,
        human=True,
        save_path='trained/ppo_last.pt'
    )
    Game(config).run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train","play_vs_simple","play_vs_ai"], default="train")
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "play_vs_simple":
        run_play_vs_simple()
    elif args.mode == "play_vs_ai":
        run_play_vs_ai()
