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
    # ql = DeepQLearning(
    #     general=False,
    #     batch_size=32,
    #     size=4,
    #     player_pos=(0, 0),
    #     num_food=1,
    #     nuke_prob= 0.5,
    #     min_epsilon=0.001,
    #     max_epsilon=0.1,
    #     intended_action_prob=1,
    #     wins_threshold=0.8
    # )
    #
    # ql.train()
    # ql.play()

    ac = OneStepActorCritic(
        general=False,
        size=4,
        player_pos=(0, 0),
        num_food=1,
        nuke_prob= 0.5,
        intended_action_prob=1,
        wins_threshold=0.9
    )

    ac.train()
    ac.play()

    results = pd.DataFrame(
        columns=[
            "win_rate",
            "episode"
        ]
    )

    outcomes = deque(maxlen=100)
    for i in range(1, 1000):
        ac.play()
        outcomes.append(1 if is_terminal(ac.environment) == "WIN" else 0)

        if len(outcomes) >= outcomes.maxlen and i % 10 == 0:
            results.loc[len(results)] = {
                "win_rate": np.mean(outcomes),
                "episode": i,
            }
        ac.environment = ac.environments.get()

    average_win_rate = results["win_rate"].mean() * 100

    plt.plot(results["episode"], results["win_rate"] * 100, color='r',
             label='Episode Win Rate')
    plt.axhline(y=average_win_rate, color='blue', linestyle='--',
                label=f'Average: {average_win_rate:.2f}%')
    plt.title('Deep Q Learning Win Rate')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate (%)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

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