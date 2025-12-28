import time
from multiprocessing import Queue, Process

import numpy as np
from PySide6.QtWidgets import QApplication

from src.game.game_rules import generate_environment, is_terminal, \
    move_with_chance
from src.game.tiles import Tiles


class Game:

    def __init__(self,
                 size: int = 5,
                 player_pos: tuple = (),
                 num_food: int = -1,
                 nuke_prob: float = 0.7,
                 intended_action_prob: float = 0.75,
                 transfer_state: dict = None,
                 ):
        self.size: int = size
        self.player_pos = player_pos
        self.num_food = num_food
        self.nuke_prob = nuke_prob
        self.intended_action_prob: float = intended_action_prob
        self.actions: tuple = ("UP", "DOWN", "LEFT", "RIGHT")

        if transfer_state:
            self.environments = transfer_state['queue']
            self.generator_process = transfer_state['process']
            self.environment = transfer_state['environment']
        else:
            self.environments = Queue(maxsize=10)
            self.generator_process = Process(
                target=environment_generator_worker, args=(
                    self.environments,
                    self.size,
                    self.actions,
                    self.player_pos,
                    self.num_food,
                    self.nuke_prob
                ))
            self.generator_process.daemon = True
            self.generator_process.start()
            self.environment = self.environments.get()

    def play(self, controller = None):
        if controller is None:
            print(f"Initial Environment: \n {self.get_styled_environment()}\n")

        i = 0
        while (result := is_terminal(self.environment)) == "":
            i += 1
            action: str = self.get_action()
            is_intended_action  = move_with_chance(self.environment, action, self.actions, self.intended_action_prob)
            if controller is not None:
                controller.update_board(self.environment)
                QApplication.processEvents()
                time.sleep(controller.move_time_dspin.value())
            else:
                print(f"{i}.\n{self.get_styled_environment()}")
                print(("Intended Move" if is_intended_action else "Unintended Move") + "\n")

        if controller is not None:
            controller.game_over(result)
        else:
            print(f"YOU {result} after {i} moves" )

    def get_action(self) -> str:
        return input("Enter a move: ")

    def get_styled_environment(self):
        return np.array([[cell.value for cell in row] for row in self.environment])

    def train(self):
        pass


def environment_generator_worker(
    queue: Queue,
    size: int,
    actions: tuple,
    player_pos: tuple,
    num_food: int,
    nuke_prob: float
):
    while True:
        new_env = generate_environment(
            size, actions, player_pos, num_food, nuke_prob
        )
        queue.put(new_env)

