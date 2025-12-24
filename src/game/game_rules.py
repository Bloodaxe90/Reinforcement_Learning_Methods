import collections
from multiprocessing import Queue
from random import shuffle

import numpy as np
import random

from src.game.tiles import Tiles


def generate_environment(size: int,
                         actions: tuple,
                         player_pos: tuple = (),
                         num_food: int = -1,
                         nuke_prob: float = 0.7
                         ) -> np.ndarray:
    assert size > 1, "Size must be at least 2"
    environment: np.ndarray = np.full((size, size), Tiles.EMPTY).astype(object)

    # Create random player position
    x_pos, y_pos = player_pos if player_pos else (random.randint(0, size -1), random.randint(0, size -1))
    environment[x_pos][y_pos] = Tiles.PLAYER

    # Add Food
    empty_pos = get_tile_pos(environment, Tiles.EMPTY)
    if num_food < 0:
        num_food = random.randint(1, len(empty_pos))
    assert num_food <= len(
        empty_pos), f"Can have no more than {len(empty_pos)} food"
    food_pos = random.sample(empty_pos,
                             k=num_food
                             )
    environment[tuple(zip(*food_pos))] = Tiles.FOOD

    # Add Nukes
    empty_pos = get_tile_pos(environment, Tiles.EMPTY)
    shuffle(empty_pos)
    for pos in empty_pos:
        tmp_environment = environment.copy()
        if random.random() <= nuke_prob:
            tmp_environment[pos] = Tiles.NUKE
            if set(food_pos).issubset(set(get_accessible_tile_pos(tmp_environment, actions, player_pos))):
                environment[pos] = Tiles.NUKE
    num_nuke = len(get_tile_pos(environment, Tiles.NUKE))

    # print(f"Environment Generated with: \n"
    #       f"  - Player at position {player_pos}\n"
    #       f"  - {num_food} pieces of Food\n"
    #       f"  - {num_nuke} nukes\n")
    return environment

def get_accessible_tile_pos(environment: np.ndarray, actions: tuple, player_pos: tuple, visited: list = None):
    if visited is None: visited = []
    if Tiles.PLAYER_NUKE is environment[player_pos] or player_pos in visited:
        return visited
    visited.append(player_pos)
    for action in actions:
        tmp_environment = environment.copy()
        if move(tmp_environment, action):
            player_pos = get_tile_pos(tmp_environment, Tiles.PLAYER)
            get_accessible_tile_pos(tmp_environment, actions, player_pos, visited)
    return visited

def move(environment: np.ndarray, action: str) -> bool:
    player_pos = get_tile_pos(environment, Tiles.PLAYER)
    i, j = player_pos
    match action:
        case "UP":
            i -= 1
        case "DOWN":
            i += 1
        case "LEFT":
            j -= 1
        case "RIGHT":
            j += 1

    if 0 <= i < len(environment) and 0 <= j < len(environment[0]):
        environment[player_pos] = Tiles.EMPTY
        place_player(environment, (i, j))
        return True
    return False

def move_with_chance(environment: np.ndarray,
                     action: str,
                     actions: tuple,
                     intended_action_prob: float) -> bool:
    if random.random() <= intended_action_prob:
        move(environment, action)
        return True
    other_actions = list(actions)
    other_actions.remove(action)
    move(environment, random.choice(other_actions))
    return False

def place_player(environment: np.ndarray, new_player_pos: tuple):
    match environment[new_player_pos]:
        case Tiles.FOOD:
            environment[new_player_pos] = Tiles.PLAYER_FOOD
        case Tiles.NUKE:
            environment[new_player_pos] = Tiles.PLAYER_NUKE
        case Tiles.EMPTY:
            environment[new_player_pos] = Tiles.PLAYER


def get_tile_pos(environment: np.ndarray, tile_type: Tiles) -> tuple or list[tuple]:
    if tile_type is Tiles.PLAYER:
        return tuple(np.argwhere((environment == Tiles.PLAYER) |
                                 (environment == Tiles.PLAYER_NUKE) |
                                 (environment == Tiles.PLAYER_FOOD)
                                 ).squeeze(0).tolist())
    return [tuple(pos) for pos in np.argwhere(environment == tile_type)]


def get_valid_action_indexes(environment: np.ndarray, actions: tuple) -> list:
    return [actions.index(action) for action in actions if move(environment.copy(), action)]

def is_terminal(environment: np.ndarray) -> str:
    if is_win(environment):
        return "WIN"
    elif is_loss(environment):
        return "LOSE"
    return ""

def is_loss(environment: np.ndarray) -> bool:
    return Tiles.PLAYER_NUKE is environment[get_tile_pos(environment, Tiles.PLAYER)]


def is_win(environment: np.ndarray) -> bool:
    return not np.any(environment == Tiles.FOOD)