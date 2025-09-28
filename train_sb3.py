import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback


def train_sb3_baseline():

    print("Creating CarRacing environment...")

    # Create vectorized environment
    env = make_vec_env("CarRacing-v3", n_envs=1, env_kwargs={"render_mode": None})

    # Create evaluation environment
    eval_env = gym.make("CarRacing-v3", render_mode=None)

    print("Initializing SB3 PPO model...")

    # Create PPO model with good hyperparameters for CarRacing
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./sb3_logs/"
    )

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./trained/",
        log_path="./eval_logs/",
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    print("Starting training...")
    print("This will train for 200k timesteps (should take 10-30 minutes)")

    # Train the model
    model.learn(
        total_timesteps=200_000,
        callback=eval_callback,
        progress_bar=True
    )

    # Save the final model
    model.save("trained/sb3_ppo_final")

    print("Training completed!")
    print("Model saved to: trained/sb3_ppo_final.zip")
    print("Best model saved to: trained/best_model.zip")

    print("\nTesting trained model...")
    obs, _ = eval_env.reset()
    total_reward = 0

    for step in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"Test episode reward: {total_reward:.2f}")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    train_sb3_baseline()