from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass
class GameConfig:
    width: int = 1600
    height: int = 1000
    fps: int = 60
    round_time: int = 1000  # milliseconds
    minimap_size: int = 180
    seed: Optional[int] = None
    number_of_cars: int = 2
    user_agent_factory: Optional[Callable[[], Any]] = None
    user_agent_training: bool = False
    render: bool = True
    human: bool = False
    save_path: str = 'trained/last.json'
