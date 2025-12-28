

import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
import os

from src.env import SixTakesEnv
from src.agents import RandomAgent, SmallChoiceAgent, LargeChoiceAgent, MaskedPPOAgent


def mask_fn(env: gym.Env):
    """Return the action mask for the current observation"""
    return env.get_wrapper_attr("action_mask")


class SelfPlayCallback(BaseCallback):
    """
    Callback for self-play training.
    Periodically evaluates the model and updates opponent models with the best weights.
    """
    def __init__(self, eval_freq: int, n_eval_episodes: int, best_model_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_path = best_model_path
        self.best_mean_reward = -float('inf')
        self.opponents = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Evaluate current model
            mean_reward = self._evaluate_model()
            
            if self.verbose > 0:
                print(f"Step {self.n_calls}: Mean reward: {mean_reward:.2f}, Best: {self.best_mean_reward:.2f}")
            
            # Save if it's the best so far
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(self.best_model_path)
                if self.verbose > 0:
                    print(f"New best model saved with mean reward: {mean_reward:.2f}")
                
                # Update opponent models
                self._update_opponents()
        
        return True
    
    def _evaluate_model(self) -> float:
        """Evaluate the current model against random opponents"""
        total_rewards = []
        
        for _ in range(self.n_eval_episodes):
            env = SixTakesEnv(opponents=[RandomAgent(), RandomAgent(), RandomAgent()])
            env = ActionMasker(env, mask_fn)
            obs, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action_mask = info["action_mask"]
                action, _ = self.model.predict(obs, action_masks=action_mask, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            
            total_rewards.append(episode_reward)
        
        return sum(total_rewards) / len(total_rewards)
    
    def _update_opponents(self):
        """Update all registered opponent models with the best weights"""
        for opponent in self.opponents:
            if isinstance(opponent, MaskedPPOAgent):
                opponent.update_model(self.best_model_path)
                if self.verbose > 0:
                    print(f"Updated opponent model")


class UpdatableMaskedPPOAgent(MaskedPPOAgent):
    """MaskedPPOAgent that can update its model during training"""
    
    def update_model(self, model_path: str):
        """Reload the model from the specified path"""
        from sb3_contrib import MaskablePPO
        self.model = MaskablePPO.load(model_path)


def make_env(opponents):
    env = SixTakesEnv(opponents=opponents)
    return ActionMasker(env, mask_fn)


if __name__ == "__main__":
    
    # Initialize with best model if it exists, otherwise start with random opponents
    best_model_path = "models/ppo_best"
    
    # Create updatable opponents
    if os.path.exists(f"{best_model_path}.zip"):
        print("Loading existing best model for opponents")
        opponents = [
            UpdatableMaskedPPOAgent(best_model_path),
            UpdatableMaskedPPOAgent(best_model_path),
            UpdatableMaskedPPOAgent(best_model_path)
        ]
    else:
        print("Starting with random opponents")
        opponents = [
            RandomAgent(),
            RandomAgent(),
            RandomAgent()
        ]
    
    # Create vectorized environments
    env = make_vec_env(lambda: make_env(opponents), n_envs=4)

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        learning_rate=3e-4,
    )

    # Create callback for self-play
    callback = SelfPlayCallback(
        eval_freq=10000,  # Evaluate every 10k steps
        n_eval_episodes=20,  # Use 20 episodes for evaluation
        best_model_path=best_model_path,
        verbose=1
    )
    
    # Register opponents with callback so they can be updated
    callback.opponents = opponents

    model.learn(total_timesteps=500000, callback=callback)

    model.save("models/ppo_v1")