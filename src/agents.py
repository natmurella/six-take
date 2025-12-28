
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional

class BaseAgent(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def policy(self, obs) -> int:
        return 0


class RandomAgent(BaseAgent):
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
    
    def policy(self, obs):
        hand = obs['hand']
        valid_indices = [i for i, card in enumerate(hand) if card != -1]
        return self.rng.choice(valid_indices)

class SmallChoiceAgent(BaseAgent):
    
    def __init__(self):
        super().__init__()
    
    def policy(self, obs):
        hand = obs['hand']
        return 0

class LargeChoiceAgent(BaseAgent):
    
    def __init__(self):
        super().__init__()
    
    def policy(self, obs):
        hand = obs['hand']
        valid_cards = [card for card in hand if card != -1]
        max_card = max(valid_cards)
        return int(valid_cards.index(max_card))