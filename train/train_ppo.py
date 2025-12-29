

import gymnasium as gym
import os
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor

from src.env import SixTakesEnv
from src.agents import RandomAgent, MaskedPPOAgent, SmallChoiceAgent


def mask_fn(env: gym.Env):
    
    return env.get_wrapper_attr("action_mask")


def make_env(use_best_model: bool = False):

    best_model_path = "models/ppo_best.zip"

    if not os.path.exists(best_model_path):
        # use random opponents initially
        opponents = [
            RandomAgent(),
            RandomAgent(),
            SmallChoiceAgent()
        ]
    else:
        scenario_idx = np.random.randint(5)

        if scenario_idx == 0:
            # use best models for full self-play
            opponents = [
                MaskedPPOAgent(best_model_path, deterministic=False),
                MaskedPPOAgent(best_model_path, deterministic=False),
                MaskedPPOAgent(best_model_path, deterministic=False)
            ]
        elif scenario_idx == 1:
            # use mix of best model and simple agents
            opponents = [
                MaskedPPOAgent(best_model_path, deterministic=False),
                SmallChoiceAgent(),
                RandomAgent()
            ]
        elif scenario_idx == 2:
            # use small simple agents only
            opponents = [
                SmallChoiceAgent(),
                SmallChoiceAgent(),
                SmallChoiceAgent()
            ]
        elif scenario_idx == 3:
            # use masked and one small agent
            opponents = [
                MaskedPPOAgent(best_model_path, deterministic=False),
                SmallChoiceAgent(),
                MaskedPPOAgent(best_model_path, deterministic=False)
            ]
        else:
            # full random
            opponents = [
                RandomAgent(),
                RandomAgent(),
                RandomAgent()
            ]

        
    
    env = SixTakesEnv(opponents=opponents)
    env = ActionMasker(env, mask_fn)
    
    return env 


def evaluate_model(model, n_episodes: int = 20):
    
    scores = []
    
    for _ in range(n_episodes):
        env = make_env(use_best_model=False)  # Eval against random
        obs, info = env.reset()
        done = False
        
        while not done:
            action_mask = info["action_mask"]
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        final_score = info["scores"][0]
        scores.append(final_score)
        env.close()
    
    return np.mean(scores)


class SelfPlayCallback(BaseCallback):
    
    def __init__(self, eval_freq: int, n_eval_episodes: int = 20, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_score = float('inf')  # Lower is better
        self.best_model_path = "models/ppo_best.zip"
    
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current model
            mean_score = evaluate_model(self.model, self.n_eval_episodes)
            
            if self.verbose > 0:
                print(f"\nEval at step {self.n_calls}: Mean score = {mean_score:.2f}")
            
            # Save if best
            if mean_score < self.best_mean_score:
                self.best_mean_score = mean_score
                self.model.save(self.best_model_path)
                if self.verbose > 0:
                    print(f"New best model saved! Score: {mean_score:.2f}")
        
        return True


if __name__ == "__main__":

    # Training parameters
    total_timesteps = 2000000
    eval_freq = 5000
    
    # Initial training against random opponents
    print("Starting training with self-play...")
    env = make_vec_env(lambda: make_env(use_best_model=False), n_envs=5)

    log_dir = "logs/train"
    os.makedirs(log_dir, exist_ok=True)

    env = VecMonitor(
        env,
        filename=os.path.join(log_dir, "monitor.csv"),
        info_keywords=("own_score", "winner", "score_diff",),
    )
    
    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
    )
    
    # Train with self-play callback
    callback = SelfPlayCallback(eval_freq=eval_freq, n_eval_episodes=20)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Save final model
    model.save("models/ppo_v1")
    print(f"\nTraining complete! Best score: {callback.best_mean_score:.2f}")
    print("Final model saved to models/ppo_v1.zip")
    print("Best model saved to models/ppo_best.zip")