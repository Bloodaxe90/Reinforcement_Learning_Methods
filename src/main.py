from src.rl_methods.model_based.policy_iteration import PolicyIteration
from src.rl_methods.model_based.value_iteration import ValueIteration
from src.rl_methods.model_free.deep_q_learning import DeepQLearning
from src.rl_methods.model_free.one_step_actor_critic import OneStepActorCritic
from src.rl_methods.model_free.reinforce import REINFORCE


def main():
    # ql = DeepQLearning(
    #     batch_size=32,
    #     size=4,
    #     player_pos=(0, 0),
    #     num_food=1,
    #     nuke_prob= 0.5,
    #     intended_action_prob=1
    # )
    #
    # ql.train()
    # ql.play()

    # ac = OneStepActorCritic(
    #     size=4,
    #     player_pos=(0, 0),
    #     num_food=1,
    #     nuke_prob= 0.5,
    #     intended_action_prob=1
    # )
    #
    # ac.train()
    # ac.play()

    # re = REINFORCE(
    #     size=4,
    #     player_pos=(0, 0),
    #     num_food=1,
    #     nuke_prob= 0.5,
    #     intended_action_prob=1
    # )
    #
    # re.train()
    # re.play()

    pi = PolicyIteration(
        size=4,
        player_pos=(0, 0),
        num_food=1,
        nuke_prob= 0.5,
        intended_action_prob=1
    )

    pi.train()
    pi.play()

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