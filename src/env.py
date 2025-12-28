from typing import Optional
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

INITIAL_HAND_SIZE = 10
MAX_CARD_VALUE = 104
NUM_STACKS = 4
MAX_STACK_SIZE = 5

class Deck:
    def __init__(self, cards, seed=None):
        self.cards = list(cards)
        self.r = random.Random(seed)
        self.r.shuffle(self.cards)
        self.idx = 0
    
    def draw(self, n=1):
        if self.idx + n > len(self.cards):
            raise ValueError("Not enough cards in deck to draw.")
        drawn = self.cards[self.idx:self.idx + n]
        self.idx += n
        return drawn


class SixTakesEnv(gym.Env):
    """Six qui prend! environment for Reinforcement Learning."""
    
    def __init__(
            self, 
            opponents=[]
            ):
        super().__init__()
        
        self.num_players = 1 + len(opponents)

        if self.num_players > 10:
            raise ValueError("Maximum number of players is 10.")
        
        # Agent chooses index of card in hand
        self.action_space = spaces.Discrete(INITIAL_HAND_SIZE)
        
        # Observation: agent hand (10 cards) + round number (1)
        self.observation_space = spaces.Dict({
            "round": spaces.Discrete(
                INITIAL_HAND_SIZE + 1
            ),
            "hand": spaces.Box(
                low=-1,
                high=MAX_CARD_VALUE,
                shape=(INITIAL_HAND_SIZE,),
                dtype=np.int32
            ),
            "stacks": spaces.Box(
                low=-1,
                high=MAX_CARD_VALUE,
                shape=(NUM_STACKS,MAX_STACK_SIZE),
                dtype=np.int32
            ),
            "stack_sizes": spaces.Box(
                low=1,
                high=MAX_STACK_SIZE,
                shape=(NUM_STACKS,),
                dtype=np.int32
            ),
            "unseen_cards": spaces.Box(
                low=0,
                high=1,
                shape=(MAX_CARD_VALUE,),
                dtype=np.int32
            ),
            "own_history": spaces.Box(
                low=-1,
                high=MAX_CARD_VALUE,
                shape=(INITIAL_HAND_SIZE,),
                dtype=np.int32
            ),
            "opponent_history": spaces.Box(
                low=-1,
                high=MAX_CARD_VALUE,
                shape=(9,INITIAL_HAND_SIZE),
                dtype=np.int32
            ),
            "own_score": spaces.Discrete(
                1
            ),
            "opponent_scores": spaces.Box(
                low=0,
                high=MAX_CARD_VALUE * INITIAL_HAND_SIZE,
                shape=(9,),
                dtype=np.int32
            )
            
        })
        
        self.opponents = opponents

        self.deck = None

        self.round = 0

        self.hands = [[] for _ in range(self.num_players)]

        self.stacks = [[] for _ in range(NUM_STACKS)]

        self.unseen_cards = set(range(1, MAX_CARD_VALUE + 1))

        self.history = [[] for _ in range(self.num_players)]

        self.scores = [0 for _ in range(self.num_players)]
    
    
    
    
    def step(self, action):

        action_idx = action

        # validate action
        if action < 0 or action >= len(self.hands[0]):
            raise ValueError("Invalid action: card index out of range.")
        
        # collect opponent actions
        opponent_action_idx = []
        for i, opponent in enumerate(self.opponents):
            obs_opp = self._get_obs(player_idx=i+1)
            opp_action_idx = opponent.policy(obs_opp)
            opponent_action_idx.append(opp_action_idx)

        card_to_player_map = {}
        # update player hand, unseen cards, history
        player_hand = self.hands[0]
        player_card = player_hand.pop(action_idx)
        self.unseen_cards.remove(player_card)
        self.history[0].append(player_card)
        card_to_player_map[player_card] = 0

        # update opponent hands, unseen cards, history
        opponent_cards = []
        for i, opp_idx in enumerate(opponent_action_idx):
            opp_hand = self.hands[i+1]
            opp_card = opp_hand.pop(opp_idx)
            self.unseen_cards.remove(opp_card)
            self.history[i+1].append(opp_card)
            card_to_player_map[opp_card] = i + 1
            opponent_cards.append(opp_card)
        
        # determine card application order
        all_played_cards = [player_card] + opponent_cards
        all_played_cards_sorted = sorted(all_played_cards)

        # apply cards to stacks (update scores)
        player_round_reward = 0
        for card in all_played_cards_sorted:
            player_idx = card_to_player_map[card]
            stack_target_idx = -1
            stack_target_top = float('-inf')
            stack_min_idx = -1
            stack_min_score = float('inf')
            stack_min_top = float('inf')
            for s_idx, stack in enumerate(self.stacks):
                stack_top = stack[-1]
                if card > stack_top and stack_top > stack_target_top:
                    stack_target_top = stack_top
                    stack_target_idx = s_idx
                stack_score = 0
                for sc in stack:
                    stack_score += self.card_to_score(sc)
                if stack_score <= stack_min_score and stack_top < stack_min_top:
                    stack_min_score = stack_score
                    stack_min_top = stack_top
                    stack_min_idx = s_idx
            
            wipe_stack = False
            if stack_target_idx == -1:
                stack_target_idx = stack_min_idx
                wipe_stack = True
            
            if len(self.stacks[stack_target_idx]) >= MAX_STACK_SIZE or wipe_stack:
                for sc in self.stacks[stack_target_idx]:
                    score_add = self.card_to_score(sc)
                    self.scores[player_idx] += score_add
                    if player_idx == 0:
                        player_round_reward += score_add
                self.stacks[stack_target_idx] = [card]
            else:
                self.stacks[stack_target_idx].append(card)

        terminated = False
        if len(self.hands[0]) == 0:
            terminated = True
        
        self.round += 1
        obs = self._get_obs()
        info = self._get_info()
        reward = -player_round_reward
        
        return obs, reward, terminated, False, info
    






    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ):
        super().reset(seed=seed)

        self.deck = Deck(
            cards=range(1, MAX_CARD_VALUE + 1),
            seed=seed
        )

        self.round = 0

        self.unseen_cards = set(range(1, MAX_CARD_VALUE + 1))

        self.hands = [[] for _ in range(self.num_players)]
        for i in range(self.num_players):
            hand = self.deck.draw(INITIAL_HAND_SIZE)
            hand = sorted(hand)
            self.hands[i] = hand

        self.stacks = [[] for _ in range(NUM_STACKS)]
        for i in range(NUM_STACKS):
            card = self.deck.draw(1)
            self.stacks[i] = card
            self.unseen_cards.remove(card[0])

        self.history = [[] for _ in range(self.num_players)]

        if options is not None and 'initial_scores' in options:
            self.scores = options['initial_scores']
        else:
            self.scores = [0 for _ in range(self.num_players)]

        obs = self._get_obs()
        info = self._get_info()

        return obs, info
    


    
    def _get_obs(self, player_idx: int = 0):

        # make padded hand
        hand = self.hands[player_idx]
        padded_hand = hand + [-1] * (INITIAL_HAND_SIZE - len(hand))

        # get stack and stack size
        stacks_sorted = sorted(
            self.stacks,
            key=lambda s: s[-1] if len(s) > 0 else -1
        )
        stack_sizes = [len(s) for s in stacks_sorted]
        stacks_sorted_padded = [
            s + [-1] * (MAX_STACK_SIZE - len(s)) for s in stacks_sorted
        ]

        # get unseen cards
        unseen_cards_array = np.zeros(MAX_CARD_VALUE, dtype=np.int32)
        for card in self.unseen_cards:
            unseen_cards_array[card - 1] = 1
        for card in hand:
            unseen_cards_array[card - 1] = 0

        # get history
        h = self.history[player_idx]
        h_padded = h + [-1] * (INITIAL_HAND_SIZE - len(h))
        opp_h = []
        for idx, opp_h_temp in enumerate(self.history):
            if idx == player_idx:
                continue
            opp_h.append(opp_h_temp)
        s_idx = sorted(
            range(len(opp_h)),
            key=lambda i: opp_h[i][-1] if len(opp_h[i]) > 0 else -1
        )
        opp_h_sorted = [opp_h[i] for i in s_idx]
        opp_h_padded = [
            hist + [-1] * (INITIAL_HAND_SIZE - len(hist)) for hist in opp_h_sorted
        ]

        # scores
        own_score = self.scores[player_idx]
        opponent_scores = []
        for idx, score in enumerate(self.scores):
            if idx == player_idx:
                continue
            opponent_scores.append(score)
        s_idx_opp_scores = [opponent_scores[i] for i in s_idx]
        
        obs = {
            "round": self.round,
            "hand": np.array(padded_hand, dtype=np.int32),
            "stacks": np.array(stacks_sorted_padded, dtype=np.int32),
            "stack_sizes": np.array(stack_sizes, dtype=np.int32),
            "unseen_cards": unseen_cards_array,
            "own_history": np.array(h_padded, dtype=np.int32),
            "opponent_history": np.array(opp_h_padded, dtype=np.int32),
            "own_score": own_score,
            "opponent_scores": np.array(s_idx_opp_scores, dtype=np.int32)
        }

        return obs
    
    def _get_info(self):
        return {}
    
    def card_to_score(self, card: int) -> int:

        if card % 55 == 0:
            return 7
        elif card % 11 == 0:
            return 5
        elif card % 10 == 0:
            return 3
        elif card % 5 == 0:
            return 2
        else:
            return 1

