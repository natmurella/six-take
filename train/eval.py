import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from stable_baselines3.common.env_util import make_vec_env

from src.env import SixTakesEnv
from src.agents import RandomAgent, MaskedPPOAgent

def mask_fn(env: gym.Env):
    return env.get_wrapper_attr("action_mask")

def make_eval_env(seed=None):
    env = SixTakesEnv(
        opponents=[
            MaskedPPOAgent(model_path="models/ppo_v1"),
            MaskedPPOAgent(model_path="models/ppo_v1"),
            MaskedPPOAgent(model_path="models/ppo_v1")
        ]
    )
    return ActionMasker(env, mask_fn)


def evaluate(
    model_path: str,
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
):
    scores = []
    placements = []
    distances = []
    total_rewards = []

    for ep in range(n_episodes):
        env = make_eval_env(seed=ep)
        obs, info = env.reset()

        done = False
        ep_reward = 0.0

        while not done:
            # Get action mask from info and pass to predict
            action_masks = info["action_mask"]
            action, _ = model.predict(obs, action_masks=action_masks, deterministic=deterministic)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += float(reward)

            if render:
                env.render()

        info_scores = info["scores"]
        final_score = info_scores[0]
        scores.append(final_score)
        total_rewards.append(ep_reward)

        scores_ordered = sorted(info_scores)
        placement = scores_ordered.index(final_score) + 1
        placements.append(placement)

        if placement == 1:
            distances.append(0)
        else:
            dist = final_score - scores_ordered[0]
            distances.append(dist)

        print(
            f"Episode {ep:03d} | "
            f"Final score: {final_score:4d} | "
            f"Total reward: {ep_reward:7.2f} | "
            f"Placement: {placement:1d} | "
            f"Distance to 1st: {distances[-1]:4d}"
        )

        env.close()

    print("\n=== Evaluation Summary ===")
    print(f"Episodes: {n_episodes}")
    print(f"Mean final score: {np.mean(scores):.2f}")
    print(f"Std final score:  {np.std(scores):.2f}")
    print(f"Best score:       {np.min(scores)}")
    print(f"Worst score:      {np.max(scores)}")
    print(f"Mean reward:      {np.mean(total_rewards):.2f}")
    print(f"Mean placement:   {np.mean(placements):.2f}")
    print(f"Mean distance to 1st: {np.mean(distances):.2f}")


    return {
        "scores": scores,
        "rewards": total_rewards,
    }


if __name__ == "__main__":
    MODEL_PATH = "models/ppo_v1"

    model = MaskablePPO.load(MODEL_PATH)

    evaluate(
        model_path=MODEL_PATH,
        n_episodes=50,
        deterministic=True,
        render=False,
    )
