import pytest
from src.env import SixTakesEnv
from src.agents import RandomAgent, SmallChoiceAgent, LargeChoiceAgent


def test_simple():
    env = SixTakesEnv(
        opponents=[
            RandomAgent(seed=42),
            SmallChoiceAgent(),
            LargeChoiceAgent()
        ]
    )

    player = RandomAgent(seed=42)

    obs, info = env.reset(seed=42)

    # check initial stacks
    assert obs["stacks"][0][0] == 13
    assert obs["stacks"][1][0] == 37
    assert obs["stacks"][2][0] == 56
    assert obs["stacks"][3][0] == 89
    assert obs["stacks"][0][1] == -1 and obs["stacks"][1][1] == -1 and obs["stacks"][2][1] == -1 and obs["stacks"][3][1] == -1

    total_reward: float = 0

    action = player.policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    # check history correct
    assert obs["own_history"][0] == 2
    assert obs["opponent_history"][0][0] == 9
    assert obs["opponent_history"][1][0] == 11
    assert obs["opponent_history"][2][0] == 103
    assert obs["opponent_history"][0][1] == -1 and obs["opponent_history"][1][1] == -1 and obs["opponent_history"][2][1] == -1

    # check hand still 10 but suffixed with -1s
    assert len(obs["hand"]) == 10 and obs["hand"][-1] == -1

    # check new stacks
    assert obs["stacks"][0][0] == 2 and obs["stacks"][0][1] == 9 and obs["stacks"][0][2] == 11
    assert obs["stacks"][1][0] == 37
    assert obs["stacks"][2][0] == 56
    assert obs["stacks"][3][0] == 89 and obs["stacks"][3][1] == 103

    # check stack sizes
    assert obs["stack_sizes"][0] == 3
    assert obs["stack_sizes"][1] == 1
    assert obs["stack_sizes"][2] == 1
    assert obs["stack_sizes"][3] == 2

    # check unseen cards (inc player hand)
    seen_cards = [2, 9, 11, 103, 13, 37, 56, 89, 10, 16, 43, 59, 66, 74, 91, 99, 104]
    for card in seen_cards:
        assert obs["unseen_cards"][card-1] == 0
    
    unseen_cards_sample = [8, 12, 44, 55, 57]
    for card in unseen_cards_sample:
        assert obs["unseen_cards"][card-1] == 1

    # check scores are correct
    assert obs["own_score"] == 1
    
    assert reward == -1

    action = player.policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    assert reward == 0

    action = player.policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    assert reward == 0

    action = player.policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    assert reward == 0

    