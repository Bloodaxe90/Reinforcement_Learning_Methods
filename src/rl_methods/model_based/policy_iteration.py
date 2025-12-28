import random

import numpy as np

from src.game.game import Game
from src.game.game_rules import is_terminal
from src.rl_methods.utils import get_states, get_state_key, get_reward_mb, \
    get_expected_value, get_state_space


class PolicyIteration(Game):

    def __init__(self,
                 gamma: float = 0.9,
                 epsilon: float = 1e-4,
                 size: int = 5,
                 player_pos: tuple = (),
                 num_food: int = -1,
                 nuke_prob: float = 0.7,
                 intended_action_prob: float = 0.75,
                 transfer_state: dict = None,
                 ):
        super().__init__(size=size,
                         player_pos=player_pos,
                         num_food=num_food,
                         nuke_prob=nuke_prob,
                         intended_action_prob=intended_action_prob,
                         transfer_state=transfer_state
                         )
        self.policy: dict = {}
        self.gamma: float = gamma
        self.epsilon: float = epsilon
        self.intended_action_prob: float = intended_action_prob


    def train(self):
        print(f"TRAINING BEGUN, State Space: {get_state_space(self.environment)}")
        states = get_states(self.environment)
        i = 0
        while True:
            i += 1
            values = self.policy_evaluation(states)

            old_policy = self.policy.copy()
            self.policy_improvement(values, states)

            if old_policy == self.policy:
                print(f"Convergence criteria met after {i} iterations\nTRAINING FINISHED\n")
                return

    def policy_evaluation(self, states: list) -> dict:
        values = {}
        while True:
            delta = 0
            for current_state in states:
                current_state_key = get_state_key(current_state)
                action = self.policy.get(current_state_key, random.choice(self.actions))
                reward = get_reward_mb(current_state, action)
                if is_terminal(current_state):
                    new_value = reward
                else:
                    new_value = get_expected_value(current_state,
                                                   self.actions,
                                                   values,
                                                   action,
                                                   reward,
                                                   self.gamma,
                                                   self.intended_action_prob)
                delta = max(delta, abs(new_value - values.get(current_state_key, 0)))
                if delta < self.epsilon:
                    return values
                values[current_state_key] = new_value


    def policy_improvement(self, values: dict, states: list):
        for current_state in states:
            if is_terminal(current_state):
                continue

            current_state_key = get_state_key(current_state)
            candidate_values = []
            for action in self.actions:
                reward = get_reward_mb(current_state, action)
                candidate_values.append(
                    get_expected_value(current_state,
                                       self.actions,
                                       values,
                                       action,
                                       reward,
                                       self.gamma,
                                       self.intended_action_prob)
                )
            self.policy[current_state_key] = self.actions[np.argmax(np.array(candidate_values))]

    def get_action(self):
        return self.policy.get(get_state_key(self.environment), random.choice(self.actions))