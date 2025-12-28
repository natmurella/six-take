

import pytest
from src.env import SixTakesEnv
from src.agents import RandomAgent, SmallChoiceAgent, LargeChoiceAgent


def test_small_choice_agent():
    agent = SmallChoiceAgent()
    obs = {'hand': [3, 5, 7, 8, -1, -1]}
    action = agent.policy(obs)
    assert action == 0 

def test_large_choice_agent():
    agent = LargeChoiceAgent()
    obs = {'hand': [3, 5, 7, 8, -1, -1]}
    action = agent.policy(obs)
    assert action == 3

    obs = {'hand': [10, 2, 15, -1, -1, -1]}
    action = agent.policy(obs)
    assert action == 2

    obs = {'hand': [1, 1, 111]}
    action = agent.policy(obs)
    assert action == 2