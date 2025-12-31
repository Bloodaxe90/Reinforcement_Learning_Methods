import time

import numpy as np
from PySide6.QtCore import QObject, Slot, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (QWidget, QLabel, QRadioButton, QSpinBox,
                               QDoubleSpinBox, QApplication, QCheckBox)

from src.game.game import Game
from src.game.game_rules import move_with_chance, is_terminal
from src.game.tiles import Tiles
from src.rl_methods.model_based.policy_iteration import PolicyIteration
from src.rl_methods.model_based.value_iteration import ValueIteration
from src.rl_methods.model_free.deep_q_learning import DeepQLearning
from src.rl_methods.model_free.one_step_actor_critic import OneStepActorCritic
from src.rl_methods.model_free.reinforce import REINFORCE


class Controller(QObject):

    def __init__(self, scene):
        super().__init__()
        self.scene = scene

        self.grid: QWidget = self.scene.findChild(QWidget, "grid")
        self.grid.setFocus()


        self.terminal_label: QLabel = self.scene.findChild(QLabel, "LTerminal")
        self.default_radio: QRadioButton = self.scene.findChild(QRadioButton,
                                                                "DefaultRadio")
        self.vi_radio: QRadioButton = self.scene.findChild(QRadioButton,
                                                           "VIRadio")
        self.pi_radio: QRadioButton = self.scene.findChild(QRadioButton,
                                                           "PIRadio")
        self.a2c_radio: QRadioButton = self.scene.findChild(QRadioButton,
                                                            "A2CRadio")
        self.dq_radio: QRadioButton = self.scene.findChild(QRadioButton,
                                                           "DQRadio")
        self.re_radio: QRadioButton = self.scene.findChild(QRadioButton,
                                                           "RERadio")

        for radio in [self.vi_radio, self.pi_radio, self.a2c_radio,
                      self.dq_radio, self.re_radio, self.default_radio]:
            radio.toggled.connect(self.radio_button_clicked)

        self.size_spin: QSpinBox = self.scene.findChild(QSpinBox, "size_spin")
        self.player_pos_x_spin: QSpinBox = self.scene.findChild(QSpinBox,
                                                                "player_pos_x_spin")
        self.player_pos_y_spin: QSpinBox = self.scene.findChild(QSpinBox,
                                                                "player_pos_y_spin")
        self.food_spin: QSpinBox = self.scene.findChild(QSpinBox, "food_spin")
        self.batch_size_spin: QSpinBox = self.scene.findChild(QSpinBox,
                                                              "batch_size_spin")
        self.buffer_capacity_spin: QSpinBox = self.scene.findChild(QSpinBox,
                                                                   "buffer_capacity_spin")
        self.main_freq_spin: QSpinBox = self.scene.findChild(QSpinBox,
                                                             "main_freq_spin")
        self.target_freq_spin: QSpinBox = self.scene.findChild(QSpinBox,
                                                               "target_freq_spin")
        self.general_check: QCheckBox = self.scene.findChild(QCheckBox,
                                                               "general_check")

        self.nuke_prob_dspin: QDoubleSpinBox = self.scene.findChild(
            QDoubleSpinBox, "nuke_prob_dspin")
        self.intended_action_dspin: QDoubleSpinBox = self.scene.findChild(
            QDoubleSpinBox, "intended_action_dspin")
        self.gamma_dspin: QDoubleSpinBox = self.scene.findChild(QDoubleSpinBox,
                                                                "gamma_dspin")
        self.move_time_dspin: QDoubleSpinBox = self.scene.findChild(QDoubleSpinBox,
                                                                "move_time_dspin")
        self.epsilon_dspin: QDoubleSpinBox = self.scene.findChild(
            QDoubleSpinBox, "epsilon_dspin")
        self.alpha_dspin: QDoubleSpinBox = self.scene.findChild(QDoubleSpinBox,
                                                                "alpha_dspin")
        self.win_threshold_dspin: QDoubleSpinBox = self.scene.findChild(
            QDoubleSpinBox, "win_threshold_dspin")
        self.epsilon_min_dspin: QDoubleSpinBox = self.scene.findChild(
            QDoubleSpinBox, "epsilon_min_dspin")
        self.epsilon_max_dspin: QDoubleSpinBox = self.scene.findChild(
            QDoubleSpinBox, "epsilon_max_dspin")
        self.critic_alpha_dspin: QDoubleSpinBox = self.scene.findChild(
            QDoubleSpinBox, "ctritic_alpha_dspin")

        self.board: list = []
        self.stop: bool = False

        self.last_full_params = None
        self.last_env_params = None
        self.last_environment = None


        self.game: Game = None
        self.update_game()
        self.last_environment = self.game.environment.copy()
        self.reset()

    @Slot()
    def key_pressed(self, event: QKeyEvent):

        if event.key() == Qt.Key_U:
            self.update_game()
        elif event.key() == Qt.Key_R:
            self.reset()
        elif event.key() == Qt.Key_Space:
            self.game.environment = self.game.environments.get()
            self.last_environment = self.game.environment.copy()
            self.reset()
            self.terminal_label.setText("NEW BOARD")


        if self.default_radio.isChecked():
            key = event.key()
            action = None
            if key == Qt.Key_Up:
                action = "UP"
            elif key == Qt.Key_Down:
                action = "DOWN"
            elif key == Qt.Key_Left:
                action = "LEFT"
            elif key == Qt.Key_Right:
                action = "RIGHT"

            if action:
                self.play(action)
        else:
            if event.key() == Qt.Key_P:
                self.terminal_label.setText("PLAYING")
                self.game.play(self)
                self.update_board(self.game.environment)
            elif event.key() == Qt.Key_T:
                if is_terminal(self.game.environment) == "":
                    self.terminal_label.setText("TRAINING")
                    self.terminal_label.repaint()
                    QApplication.processEvents()
                    self.game.train()
                    self.update_board(self.game.environment)
                    self.terminal_label.setText("COMPLETE")


    def play(self, action: str):
        self.terminal_label.setText("PLAYING")
        move_with_chance(self.game.environment,
                         action,
                         self.game.actions,
                         self.game.intended_action_prob)
        self.update_board(self.game.environment)

        result = is_terminal(self.game.environment)
        if result:
            self.game_over(result)

    def update_board(self, environment: np.ndarray):
        current_size = self.game.size
        layout = self.grid.layout()

        need_rebuild = False
        if not self.board:
            need_rebuild = True
        elif len(self.board) != current_size:
            need_rebuild = True

        if need_rebuild:
            if self.board:
                for row in self.board:
                    for label in row:
                        layout.removeWidget(label)
                        label.deleteLater()

            self.board = []
            for row in range(current_size):
                row_tiles = []
                for col in range(current_size):
                    new_label = QLabel()
                    new_label.setScaledContents(True)
                    layout.addWidget(new_label, row, col)
                    row_tiles.append(new_label)
                self.board.append(row_tiles)

        for row in range(current_size):
            for column in range(current_size):
                self.board[row][column].setStyleSheet(
                    f"background-color: {Tiles(environment[row][column]).colour.name()};"
                )


    def create_game_instance(self, game_class):
        player_pos = ()
        if self.player_pos_x_spin.value() >= 0 and self.player_pos_y_spin.value() >= 0:
            player_pos = (
            self.player_pos_x_spin.value(), self.player_pos_y_spin.value())

        env_params = {
            'size': self.size_spin.value(),
            'player_pos': player_pos,
            'num_food': self.food_spin.value(),
            'nuke_prob': self.nuke_prob_dspin.value(),
            'intended_action_prob': self.intended_action_dspin.value(),
        }
        other_params = {}
        if game_class is not Game:
            other_params = {'gamma': self.gamma_dspin.value()}

        if game_class in [ValueIteration, PolicyIteration]:
            other_params['epsilon'] = self.epsilon_dspin.value()
        elif game_class == DeepQLearning:
            other_params.update({
                'general': self.general_check.isChecked(),
                'alpha': self.alpha_dspin.value(),
                'wins_threshold': self.win_threshold_dspin.value(),
                'batch_size': self.batch_size_spin.value(),
                'replay_capacity': self.buffer_capacity_spin.value(),
                'main_update_freq': self.main_freq_spin.value(),
                'target_update_freq': self.target_freq_spin.value(),
                'min_epsilon': self.epsilon_min_dspin.value(),
                'max_epsilon': self.epsilon_max_dspin.value(),
            })
        elif game_class == REINFORCE:
            other_params.update({
                'general': self.general_check.isChecked(),
                'alpha': self.alpha_dspin.value(),
                'wins_threshold': self.win_threshold_dspin.value()
            })
        elif game_class == OneStepActorCritic:
            other_params.update({
                'general': self.general_check.isChecked(),
                'actor_alpha': self.alpha_dspin.value(),
                'critic_alpha': self.critic_alpha_dspin.value(),
                'wins_threshold': self.win_threshold_dspin.value()
            })

        full_params = {**env_params, **other_params}

        if self.last_env_params == env_params:

            transfer_data = {
                'queue': self.game.environments,
                'process': self.game.generator_process,
                'environment': self.game.environment
            }

            new_game = game_class(**full_params,
                                  transfer_state=transfer_data)
            self.last_full_params = full_params

            return new_game

        elif self.game is not None:

            if self.game.generator_process.is_alive():
                self.game.generator_process.terminate()
                self.game.generator_process.join()

        self.last_env_params = env_params
        self.last_full_params = full_params

        return game_class(**full_params)

    @Slot()
    def radio_button_clicked(self, checked: bool):
        if not checked:
            return
        self.update_game()

    def update_game(self):
        self.terminal_label.setText("UPDATED")
        target_class = Game

        if self.vi_radio.isChecked():
            target_class = ValueIteration
        elif self.pi_radio.isChecked():
            target_class = PolicyIteration
        elif self.dq_radio.isChecked():
            target_class = DeepQLearning
        elif self.re_radio.isChecked():
            target_class = REINFORCE
        elif self.a2c_radio.isChecked():
            target_class = OneStepActorCritic

        self.game = self.create_game_instance(target_class)
        self.update_board(self.game.environment)

    def game_over(self, result: str):
        self.stop = True
        if result == "WIN":
            self.terminal_label.setStyleSheet(
                "color: green;")
        else:
            self.terminal_label.setStyleSheet(
                "color: red;")
        self.terminal_label.setText(result)

    def reset(self):
        self.game.environment = self.last_environment.copy()
        self.stop = False
        self.terminal_label.setText("RESET")
        self.terminal_label.setStyleSheet(
            "color: orange;")
        self.update_board(self.game.environment)