from collections import deque

import numpy as np

from src.game.game import Game
from src.game.game_rules import move_with_chance, is_terminal, \
    get_valid_action_indexes
from src.game.tiles import Tiles
from src.models.cnn import CNN
from src.rl_methods.utils import get_one_hot, get_reward_mf
import torch

class REINFORCE(Game):

    def __init__(self,
                 general: bool = False,
                 gamma: float = 0.9,
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
        self.policy_network: CNN = CNN(input_channels= len(Tiles) - 2,
                                  output_neurons= len(self.actions),
                                  state_size= size,
                                  max_output_channels= 16,
                                  drop_prob= 0.2)
        # summary(self.policy_network, (1, len(Tiles) - 2, size, size))
        self.general = general
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(params=self.policy_network.parameters(), lr= alpha)
        self.wins_threshold = wins_threshold


    def train(self):
        print(f"TRAINING BEGUN, general: {self.general}")
        outcomes = deque(maxlen=100)
        self.policy_network.train()
        i = 1
        while True:
            state = self.environments.get() if self.general else self.environment.copy()
            rewards = []
            action_probs = []
            visited = {state.tobytes()}
            while (outcome := is_terminal(state)) == "":
                self.play_step(state, action_probs, rewards, visited)

            self.update_network(action_probs, rewards)

            outcomes.append(1 if outcome == "WIN" else 0)
            win_rate = np.mean(outcomes)
            if len(outcomes) >= outcomes.maxlen and win_rate >= self.wins_threshold:
                print(f"Convergence criteria met after {i} iterations\nTRAINING FINISHED\n")
                break
            print(f"Iteration {i}, {outcome}, percent {win_rate}")

            i += 1

    def play_step(self, state: np.ndarray, action_probs: list, rewards: list, visited: set):
        one_hot_state = get_one_hot(state)
        logits = self.policy_network(one_hot_state.unsqueeze(0)).squeeze(0)

        mask = torch.full_like(logits, -1e9)
        for idx in get_valid_action_indexes(state, self.actions):
            mask[idx] = 0.0

        y_pred = torch.softmax((logits + mask), dim=0)
        action_index = torch.multinomial(y_pred, 1).item()
        action = self.actions[action_index]

        rewards.append(get_reward_mf(state, action, visited))
        action_probs.append(y_pred[action_index])

        move_with_chance(state, action, self.actions,
                         self.intended_action_prob)

        visited.add(state.tobytes())


    def update_network(self, action_probs, rewards):
        actual_returns = []
        discounted_reward = 0
        for reward in reversed(rewards):
            discounted_reward = reward + (self.gamma * discounted_reward)
            actual_returns.insert(0, discounted_reward)

        actual_returns = torch.tensor(actual_returns, dtype=torch.float32)
        action_probs = torch.stack(action_probs)

        loss = -torch.mean(torch.log(action_probs + 1e-9) * actual_returns)
        self.policy_network.zero_grad()
        loss.backward()
        self.optimizer.step()


    def play(self, controller = None):
        self.policy_network.eval()
        with torch.inference_mode():
            super().play(controller)

    def get_action(self) -> str:
        logits = self.policy_network(get_one_hot(self.environment).unsqueeze(0)).squeeze()
        mask = torch.full_like(logits, -1e9)
        for idx in get_valid_action_indexes(self.environment, self.actions):
            mask[idx] = 0.0

        return self.actions[
            torch.argmax(
                torch.softmax(logits + mask, dim = 0)
            , dim = 0)
        ]