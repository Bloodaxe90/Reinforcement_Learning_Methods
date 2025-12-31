import math
import random
from collections import deque

import numpy as np
import pandas as pd
import torch
from torchinfo import summary
from matplotlib import pyplot as plt

from src.game.game import Game
from src.game.game_rules import is_terminal, move_with_chance, \
    get_valid_action_indexes, get_tile_pos
from src.game.tiles import Tiles
from src.models.cnn import CNN
from src.rl_methods.utils import get_one_hot, get_reward_mf

class DeepQLearning(Game):

    def __init__(self,
                 general: bool = False,
                 batch_size: int = 32,
                 replay_capacity: int = 100,
                 main_update_freq: int = 1,
                 target_update_freq: int = 20,
                 min_epsilon: float = 0.01,
                 max_epsilon: float = 0.25,
                 gamma: float = 0.90,
                 alpha: float = 0.0001,
                 wins_threshold: float = 0.75,
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
        self.gamma = gamma
        self.main_network: CNN = CNN(input_channels=len(Tiles) - 2,
                                     output_neurons=len(self.actions),
                                     state_size=size,
                                     max_output_channels=16,
                                     drop_prob=0.2)
        self.optimizer = torch.optim.Adam(
            params=self.main_network.parameters(), lr=alpha)
        # summary(self.main_network, (1, len(Tiles) - 2, size, size))

        # Must have same architecture as main_network
        self.target_network: CNN = CNN(input_channels=len(Tiles) - 2,
                                       output_neurons=len(self.actions),
                                       state_size=size,
                                       max_output_channels=16,
                                       drop_prob=0.2)
        assert replay_capacity >= batch_size, "Replay Buffer max capacity must be greater than or equal to batch size"
        self.general = general
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.wins_threshold = wins_threshold
        self.batch_size = batch_size
        self.main_update_freq = main_update_freq
        self.target_update_freq = target_update_freq
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.epsilon = max_epsilon

    def train(self):
        print(f"TRAINING BEGUN, general: {self.general}")
        outcomes = deque(maxlen=100)
        self.main_network.train()
        self.epsilon = self.max_epsilon
        self.replay_buffer.clear()
        i = 1
        while True:
            state = self.environments.get() if self.general else self.environment.copy()
            visited = {state.tobytes()}
            while (outcome := is_terminal(state)) == "":
                self.play_step(state, visited)
                if len(self.replay_buffer) >= self.replay_buffer.maxlen:
                    if i % self.main_update_freq == 0:
                        self.update_main_network()
                    if i % self.target_update_freq == 0:
                        self.update_target_network()

            outcomes.append(1 if outcome == "WIN" else 0)
            win_rate = float(np.mean(outcomes))
            self.decay_epsilon(win_rate)
            if len(outcomes) >= outcomes.maxlen and win_rate >= self.wins_threshold:
                print(
                    f"Convergence criteria met after {i} iterations\nTRAINING FINISHED\n")
                break
            print(f"Iteration {i}, {outcome}, percent {win_rate}, epsilon {self.epsilon}")

            i += 1

    def play_step(self, state: np.ndarray, visited: set):
        current_one_hot_state = get_one_hot(state).unsqueeze(0)
        q_values = self.main_network(current_one_hot_state).squeeze(
            0)
        valid_action_indexes = get_valid_action_indexes(state, self.actions)
        action = self.get_best_action(q_values, valid_action_indexes) if random.random() >= self.epsilon else self.actions[random.choice(valid_action_indexes)]
        reward = get_reward_mf(current_state=state,
                               action=action,
                               visited=visited,
                               win_reward=10,
                               loss_reward=-10,
                               intermediate_reward=1,
                               general_reward=-1,
                               explore_reward=0)
        move_with_chance(state, action, self.actions,
                         self.intended_action_prob)
        visited.add(state.tobytes())
        self.replay_buffer.append(
            (current_one_hot_state, self.actions.index(action), reward, state, is_terminal(state) != ""))

    def update_main_network(self):
        current_one_hot_states, action_indexes, rewards, next_states, terminals = zip(*random.sample(
            self.replay_buffer, self.batch_size))

        action_indexes = torch.tensor(action_indexes, dtype=torch.int64)
        current_q_values = self.main_network(torch.cat(current_one_hot_states)).gather(1, action_indexes.unsqueeze(-1)).squeeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_q_values = torch.amax(self.target_network(torch.cat([get_one_hot(state).unsqueeze(0) for state in next_states])), dim = 1)
        terminals = torch.tensor(terminals, dtype= torch.float32)

        td_errors = rewards + (
                 (1 - terminals) * self.gamma * next_q_values) - current_q_values

        loss = torch.mean(td_errors ** 2, dim=0)
        self.main_network.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.main_network.state_dict())

    def get_best_action(self, q_values: torch.Tensor, valid_actions: list) -> str:
        mask = torch.full_like(q_values, -1e9)
        for idx in valid_actions:
            mask[idx] = 0.0

        action_index = torch.argmax((q_values + mask), 0)
        return self.actions[action_index]

    # def decay_epsilon(self, win_rate: float, min_decay_rate = 0.9999, max_decay_rate = 0.995):
    #     current_decay_rate = min_decay_rate + (
    #                 max_decay_rate - min_decay_rate) * win_rate
    #     self.epsilon *= current_decay_rate
    #     self.epsilon = max(self.min_epsilon, self.epsilon)

    def decay_epsilon(self, win_rate: float):
        decay_factor = win_rate / self.wins_threshold

        self.epsilon = self.max_epsilon - (
                    self.max_epsilon - self.min_epsilon) * decay_factor
        self.epsilon = max(self.min_epsilon, self.epsilon)
    #
    # def decay_epsilon(self, win_rate: float, power: float = 0.5):
    #     progress = min(1.0, win_rate / self.wins_threshold)
    #
    #     decay_factor = 1.0 - (progress ** power)
    #
    #     self.epsilon = self.min_epsilon + (
    #                 self.max_epsilon - self.min_epsilon) * decay_factor

    # def decay_epsilon(self, episode: int, cycle_length: int = 2000):
    #     radians = 2 * math.pi * (episode / cycle_length)
    #     wave_height = 0.5 * (1 + math.cos(radians))
    #     self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * wave_height


    def play(self, controller = None):
        self.main_network.eval()
        with torch.inference_mode():
            super().play(controller)

    def get_action(self) -> str:
        return self.get_best_action(
            self.main_network(get_one_hot(self.environment).unsqueeze(0)).squeeze(),
            get_valid_action_indexes(self.environment, self.actions)
        )