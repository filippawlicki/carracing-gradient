from multi_car_racing.Game import Game
from multi_car_racing.GameConfig import GameConfig


def main():
    config = GameConfig()
    Game(config).run()


if __name__ == "__main__":
    main()
