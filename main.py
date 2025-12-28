
from src.env import *
from src.agents import *


if __name__ == "__main__":
    env = SixTakesEnv(
        opponents=[
            RandomAgent(seed=42),
            SmallChoiceAgent(),
            LargeChoiceAgent()
        ]
    )

    player = RandomAgent(seed=42)
    
    print("Testing CardGameEnv...")
    obs, info = env.reset(seed=42)
    print(f"Initial observation: {obs}")
    
    total_reward: float = 0
    action = env.action_space.sample()
    
    for step_num in range(INITIAL_HAND_SIZE):
        action = player.policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        # Random action
        total_reward += reward
        
        print(f"Step {step_num + 1}: action={action}, reward={reward}, terminated={terminated}")
        
        if terminated:
            break
    
    print(f"\nTotal reward: {total_reward}")
    print("Environment works correctly!")