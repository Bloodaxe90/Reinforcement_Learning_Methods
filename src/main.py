from collections import deque

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.game.game_rules import is_terminal
from src.game.tiles import Tiles
from src.rl_methods.model_free.deep_q_learning import DeepQLearning
from src.rl_methods.model_free.one_step_actor_critic import OneStepActorCritic
from src.rl_methods.model_free.reinforce import REINFORCE


def main():
    ql = DeepQLearning(
        general=False,
        batch_size=32,
        size=4,
        player_pos=(0, 0),
        num_food=1,
        nuke_prob= 0.5,
        min_epsilon=0.001,
        max_epsilon=0.1,
        intended_action_prob=1,
        wins_threshold=0.8
    )

    ql.train()
    ql.play()
    # results = pd.DataFrame(
    #     columns=[
    #         "win_rate",
    #         "episode"
    #     ]
    # )
    #
    # outcomes = deque(maxlen=100)
    # for i in range(1, 1000):
    #     ql.play()
    #     outcomes.append(is_terminal(ql.environment))
    #
    #     if len(outcomes) >= outcomes.maxlen and i % 10 == 0:
    #         results.loc[len(results)] = {
    #             "win_rate": np.mean(outcomes),
    #             "episode": i,
    #         }
    #     ql.environment = ql.environments.get()
    #
    # plt.plot(results["episode"], results["win_rate"] * 100, color='r')
    # plt.title('Deep Q Learning Win Rate')
    # plt.xlabel('Episode')
    # plt.ylabel('Win Rate (%)')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # ac = OneStepActorCritic(
    #     general=False,
    #     size=4,
    #     player_pos=(0, 0),
    #     num_food=1,
    #     nuke_prob= 0.5,
    #     intended_action_prob=1,
    #     wins_threshold=0.8
    # )
    #
    # ac.train()
    # ac.play()

    # re = REINFORCE(
    #         general = False,
    #         size = 4,
    #         player_pos = (0, 0),
    #         num_food = 1,
    #         nuke_prob = 0.5,
    #         intended_action_prob = 1,
    #         wins_threshold = 0.8
    # )
    #
    # re.train()
    # re.play()

    # pi = PolicyIteration(
    #     size=4,
    #     player_pos=(0, 0),
    #     num_food=1,
    #     nuke_prob= 0.5,
    #     intended_action_prob=1
    # )
    #
    # pi.train()
    # pi.play()

    # vi = ValueIteration(size = 4,
    #                     player_pos=(0, 0),
    #                     num_food=3,
    #                     nuke_prob= 0.5,
    #                     intended_action_prob=1.0
    #                     )
    # vi.train()
    # vi.play()

    # x = Default(size = 4,
    #           player_pos=(0, 0),
    #           num_food=6,
    #           nuke_prob= 0.7)
    # x.play()

if __name__ == "__main__":
    main()