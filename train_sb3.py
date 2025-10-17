import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import os


def make_env():
    env = gym.make("CarRacing-v3", render_mode=None, continuous=True)
    env = Monitor(env)
    return env


def train_sb3_baseline(total_timesteps=500_000, use_frame_stack=False):
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using MPS")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA")
    else:
        device = "cpu"
        print(f"Using CPU")

    env = DummyVecEnv([make_env])

    if use_frame_stack:
        env = VecFrameStack(env, n_stack=4)
        print("Using frame stacking (4 frames)")

    eval_env = DummyVecEnv([make_env])
    if use_frame_stack:
        eval_env = VecFrameStack(eval_env, n_stack=4)

    os.makedirs("trained/checkpoints", exist_ok=True)
    os.makedirs("logs/sb3_ppo", exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./trained/checkpoints/",
        name_prefix="sb3_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./trained/",
        log_path="./logs/sb3_ppo/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
    )

    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=4,
        target_kl=None,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            log_std_init=-2.0,
        ),
        device=device,
        verbose=1,
    )

    print(f"Starting training for {total_timesteps} timesteps...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
        log_interval=10,
    )

    model.save("trained/sb3_ppo")
    print("Training complete! Model saved to trained/sb3_ppo.zip")

    # Test the trained model
    print("\nTesting trained model...")
    obs = env.reset()
    total_reward = 0
    steps = 0
    done = False

    while not done and steps < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]
        steps += 1

        if done:
            break

    print(f"Test episode reward: {total_reward:.2f} in {steps} steps")
    env.close()
    eval_env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--frame-stack", action="store_true",
                        help="Use frame stacking (4 frames)")

    args = parser.parse_args()

    train_sb3_baseline(total_timesteps=args.timesteps, use_frame_stack=args.frame_stack)