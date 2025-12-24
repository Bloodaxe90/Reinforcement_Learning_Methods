import numpy as np

from src.game.game import Game
from src.game.game_rules import is_terminal, move
from src.rl_methods.utils import (get_states, get_state_key, get_reward_mb
, get_expected_value, get_next_state_values, get_state_space)

class ValueIteration(Game):

    def __init__(self,
                 gamma: float = 0.9,
                 epsilon: float = 1e-4,
                 size: int = 5,
                 player_pos: tuple = (),
                 num_food: int = -1,
                 nuke_prob: float = 0.7,
                 intended_action_prob: float = 0.75, ):
        super().__init__(size=size,
                         player_pos=player_pos,
                         num_food=num_food,
                         nuke_prob=nuke_prob,
                         intended_action_prob=intended_action_prob
                         )
        self.values = {}
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.intended_action_prob: float = intended_action_prob
        self.states = get_states(self.environment)


    def train(self):
        print(f"TRAINING BEGUN, State Space: {get_state_space(self.environment)}")
        states = get_states(self.environment)
        i = 0
        while True:
            i += 1
            delta = 0
            for current_state in states:
                current_state_key = get_state_key(current_state)
                best_value = -float("inf")
                for action in self.actions:
                    reward = get_reward_mb(current_state, action)
                    if is_terminal(current_state):
                        best_value = reward
                        break
                    best_value = max(
                        best_value, get_expected_value(current_state,
                                                       self.actions,
                                                       self.values,
                                                       action,
                                                       reward,
                                                       self.gamma,
                                                       self.intended_action_prob)
                    )
                delta = max(delta, abs(best_value - self.values.get(current_state_key, 0)))
                if delta < self.epsilon:
                    print(f"Convergence criteria met after {i} iterations\nTRAINING FINISHED\n")
                    return
                self.values[current_state_key] = best_value

    def get_action(self) -> str:
        return self.actions[np.argmax(np.array(get_next_state_values(self.environment,
                              self.actions,
                              self.values)))]