from dataclasses import dataclass
from typing import Optional, Callable, Any


@dataclass
class GameConfig:
    width: int = 1400
    height: int = 900
    fps: int = 60
    minimap_size: int = 180
    seed: Optional[int] = None
    number_of_cars: int = 2
    user_agent_factory: Optional[Callable[[], Any]] = None
    user_agent_training: bool = False
    render: bool = True
    human: bool = False
    save_path: str = 'trained/last.json'
