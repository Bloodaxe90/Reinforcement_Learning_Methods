import random
from itertools import combinations
import numpy as np
import torch

from src.game.game_rules import get_tile_pos, move, place_player, is_loss, \
    is_win
from src.game.tiles import Tiles

def get_states(environment: np.ndarray) -> list:
    states: list = []
    player_pos = get_tile_pos(environment, Tiles.PLAYER)
    for food_partial_state in get_partial_states(environment, Tiles.FOOD, include_empty= False):
        for i in range(len(food_partial_state)):
            for j in range(len(food_partial_state[i])):
                state = food_partial_state.copy()
                if player_pos != (new_player_pos := (i, j)):
                    state[player_pos] = Tiles.EMPTY
                    place_player(state, new_player_pos)
                states.append(state)

    return states

def get_partial_states(environment: np.ndarray, tile: Tiles, include_empty: bool = True) -> list:
    partial_states: list = [environment]
    tile_pos = get_tile_pos(environment, tile)
    num_tiles = len(get_tile_pos(environment, tile)) + 1 if include_empty else len(get_tile_pos(environment, tile))
    for num_remove in range(1, num_tiles):
        for remove_pos in combinations(tile_pos, num_remove):
            state = environment.copy()
            for i, j in remove_pos:
                state[i][j] = Tiles.EMPTY
            partial_states.append(state)

    return partial_states

def get_state_space(state: np.ndarray) -> int:
    #dependent on get_states and get_partial_states being unchanged
    return (2 ** len(get_tile_pos(state, Tiles.FOOD)) - 1) * (len(state) * len(state[0]))


def get_expected_value(current_state: np.ndarray,
                       actions: tuple,
                       values: dict,
                       action: str,
                       reward: int,
                       gamma: float,
                       intended_action_prob: float):
    return reward + (gamma * sum(
                [prob * val for prob, val in
                 zip(
                     get_transition_probabilities(action, actions,
                                                  intended_action_prob),
                     get_next_state_values(current_state, actions, values))
                 ]
            ))

def get_reward_mb(current_state: np.ndarray,
                  action: str,
                  win_reward: int = 10,
                  loss_reward: int = -10,
                  general_reward: int = -1,
                  invalid_move_reward: int = -2) -> int:

    state_copy = current_state.copy()
    if is_win(current_state):
        return win_reward
    elif is_loss(current_state):
        return loss_reward
    if move(state_copy, action):
        return general_reward
    return invalid_move_reward

def get_reward_mf(current_state: np.ndarray,
                action: str,
                visited,
                win_reward: int = 100,
                loss_reward: int = 0,
                intermediate_reward = 70,
                general_reward: int = -1,
                explore_reward: int = 3) -> int:

    state_copy = current_state.copy()
    move(state_copy, action)
    if is_win(state_copy):
        return win_reward
    elif is_loss(state_copy):
        return loss_reward
    elif Tiles.PLAYER_FOOD is state_copy[
        get_tile_pos(state_copy, Tiles.PLAYER)]:
        return intermediate_reward
    elif state_copy.tobytes() not in visited:
        return explore_reward
    return general_reward

def get_transition_probabilities(intended_action: str,
                                 actions: tuple,
                                 intended_action_prob: float
                                 ) -> list:
    return [
        intended_action_prob
        if intended_action == action
        else (1 - intended_action_prob) / (len(actions) - 1)
        for action in actions
    ]


def get_next_state_values(current_state: np.ndarray,
                          actions: tuple,
                          values: dict,
                          ) -> list:
    possible_next_state_values = []
    for action in actions:
        possible_next_state = current_state.copy()
        move(possible_next_state, action)
        possible_next_state_values.append(
            values.get(get_state_key(possible_next_state), random.random())
        )
    return possible_next_state_values

def get_state_key(environment: np.ndarray) -> str:
    return "_".join([tile.value for tile in environment.flatten()])

def get_one_hot(environment: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.stack([
        (environment == Tiles.EMPTY),
        (
                (environment == Tiles.PLAYER) |
                (environment == Tiles.PLAYER_NUKE) |
                (environment == Tiles.PLAYER_FOOD)
        ),
        (environment == Tiles.FOOD),
        (environment == Tiles.NUKE)
    ], axis = 0)).to(torch.float32)
