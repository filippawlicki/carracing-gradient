from dataclasses import dataclass
from typing import Optional


@dataclass
class GameConfig:
    width: int = 1400
    height: int = 900
    fps: int = 60
    minimap_size: int = 180
    seed: Optional[int] = None
    number_of_cars: int = 2
