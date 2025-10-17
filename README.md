# CarRacing - Human vs AI
#### by Gradient Science Club

---

Multi-agent car racing environment with human and RL agents, training utilities and simple playback.

This repository contains a lightweight, pygame-based multi-car racing sandbox with:

- Human play mode (keyboard controls)
- Custom PPO controller and a Stable-Baselines3 (SB3) wrapper
- Training scripts and example saved models
- Optional rendering and fullscreen support

---

## Features

- Multiple agents / viewports (split-screen)
- Human vs AI, AI vs AI and training modes
- Basic on-screen HUD and minimap
- Configurable rendering and fullscreen support via `GameConfig`

---

## Requirements

- Python 3.8+
- The Python dependencies are listed in `requirements.txt`.

Install dependencies in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Quick start — run/play

The main entry point is `main.py`. A few common modes are supported; see the script's CLI for available options.

Play with human controls (rendering enabled):

```bash
python main.py --mode play_vs_ai
```

Train or run the SB3 training helper:

```bash
python main.py --mode train_sb3
```

If you prefer running the bundled training helper for the custom PPO controller, use:

```bash
python main.py --mode train
```

Watch a saved model (if available under `trained/`):

```bash
python main.py --mode watch_custom
```

---

## Configuration

The runtime configuration lives in `multi_car_racing/GameConfig.py` as a dataclass-like object. Important fields used by the code:

- `number_of_cars`: integer, how many agent viewports to create
- `render`: bool, whether pygame rendering is enabled
- `save_path`: path used to save/load trained models

You can create and pass a `GameConfig` instance in `main.py` before starting the `Game`.

---

## Controls

- WAD / Arrow keys — drive
- R — reset
- ESC — quit

---

## Project layout

- `main.py` — CLI entry and high-level mode handling
- `train_sb3.py` — helper script to train an SB3 PPO agent
- `multi_car_racing/Game.py` — main game loop, rendering and input handling
- `multi_car_racing/GameConfig.py` — configuration object
- `multi_car_racing/controllers/` — agent controllers (HumanController, PPOController, SB3Controller)
- `multi_car_racing/envs/` — environment classes (EnvCar)
- `trained/` — expected default directory for saved models (example: `trained/sb3_ppo.zip`, `trained/ppo_last.pt`)

---

## Troubleshooting

- Missing models: if playback or AI modes fail to find a model, check `trained/` for saved files or run the training mode to create them.
- Pygame errors: ensure a valid display is available (headless environments may require a virtual display).

---

## Contributing

Contributions are welcome. Small improvements that help reproducibility are especially useful (README updates, smaller test harnesses for controllers, clearer config defaults).
