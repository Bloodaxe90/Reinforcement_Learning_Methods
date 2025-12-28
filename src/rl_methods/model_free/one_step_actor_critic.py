from collections import deque
from multiprocessing import Queue, Process

import numpy as np
import torch
from torchinfo import summary

from src.game.game import Game
from src.game.game_rules import is_terminal, move_with_chance, \
    get_valid_action_indexes, generate_environment
from src.game.tiles import Tiles
from src.models.cnn import CNN
from src.rl_methods.utils import get_one_hot, get_reward_mf


class OneStepActorCritic(Game):

    def __init__(self,
                 general: bool = False,
                 gamma: float = 0.90,
                 actor_alpha: float = 0.0001,
                 critic_alpha: float = 0.0005,
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
        self.actor_network: CNN = CNN(input_channels= len(Tiles) - 2,
                                  output_neurons= len(self.actions),
                                  state_size= size,
                                  max_output_channels= 16,
                                  drop_prob= 0.2)
        self.actor_optimizer = torch.optim.Adam(params=self.actor_network.parameters(), lr= actor_alpha)
        # summary(self.actor_network, (1, len(Tiles) - 2, size, size))

        self.critic_network: CNN = CNN(input_channels=len(Tiles) - 2,
                                      output_neurons=1,
                                      state_size=size,
                                      max_output_channels=16,
                                      drop_prob=0.2)
        self.critic_optimizer = torch.optim.Adam(params=self.critic_network.parameters(), lr= critic_alpha)
        # summary(self.critic_network, (1, len(Tiles) - 2, size, size))
        self.wins_threshold = wins_threshold
        self.general = general

    def train(self):
        print(f"TRAINING BEGUN, general: {self.general}")
        outcomes = deque(maxlen=100)
        self.actor_network.train()
        self.critic_network.train()
        i = 1
        while True:
            state = self.environments.get() if self.general else self.environment.copy()
            visited = {state.tobytes()}
            while (outcome:= is_terminal(state)) == "":
                experience = self.play_step(state, visited)

                self.update_networks(experience)

            outcomes.append(1 if outcome == "WIN" else 0)
            win_rate = np.mean(outcomes)
            if len(outcomes) >= outcomes.maxlen and win_rate >= self.wins_threshold:
                print(f"Convergence criteria met after {i} iterations\nTRAINING FINISHED\n")
                break
            print(f"Iteration {i}, {outcome}, percent {win_rate}")

            i += 1

    def play_step(self, state: np.ndarray, visited: set) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        one_hot_state = get_one_hot(state)
        current_state_value = self.critic_network(one_hot_state.unsqueeze(0)).squeeze()
        logits = self.actor_network(one_hot_state.unsqueeze(0)).squeeze(0)
        mask = torch.full_like(logits, -1e9)
        for idx in get_valid_action_indexes(state, self.actions):
            mask[idx] = 0.0

        y_pred = torch.softmax((logits + mask), dim=0)
        action_index = torch.multinomial(y_pred, 1).item()
        action = self.actions[action_index]

        reward = get_reward_mf(state, action, visited)
        action_prob = y_pred[action_index]

        move_with_chance(state, action, self.actions,
                         self.intended_action_prob)
        visited.add(state.tobytes())
        one_hot_state = get_one_hot(state)
        next_state_value = self.critic_network(one_hot_state.unsqueeze(0)).squeeze() if is_terminal(state) == "" else 0
        return current_state_value, reward, action_prob, next_state_value

    def update_networks(self, experience: tuple):
        current_state_values, rewards, action_probs, next_state_values = experience
        rewards = torch.tensor(rewards, dtype= torch.float32)

        td_error_advantages = rewards + (
                    self.gamma * next_state_values) - current_state_values

        critic_loss = torch.mean(td_error_advantages ** 2, dim=0)
        self.critic_network.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -torch.mean(
            (torch.log(action_probs + 1e-9) * td_error_advantages.detach()), dim=0)
        self.actor_network.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def play(self, controller = None):
        self.actor_network.eval()
        with torch.inference_mode():
            super().play(controller)

    def get_action(self) -> str:
        logits = self.actor_network(get_one_hot(self.environment).unsqueeze(0)).squeeze()
        mask = torch.full_like(logits, -1e9)
        for idx in get_valid_action_indexes(self.environment, self.actions):
            mask[idx] = 0.0

        return self.actions[
            torch.argmax(
                torch.softmax(logits + mask, dim = 0)
            , dim = 0)
        ]




