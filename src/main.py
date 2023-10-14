from controlling.controller import Controller
from enums.opponent_type import OpponentType

if __name__ == '__main__':
    controller = Controller()
    controller.set_game_to_connect_4()
    # todo: new conda env

    # todo: refactor mcts
    # todo: refactor node
    # todo: refactor replay buffer
    # todo: refactor alpha zero agent
    # todo: refactor alpha zero algorithm

    # todo: assert alpha 0 exists for a given version
    # todo: make training alpha 0 in run

    # todo: gui gets base class
    # todo: gui errors
    # todo: why does the gui have to be refreshed in a loop

    controller.play_game(opponent_type=OpponentType.MONTE_CARLO_TREE_SEARCH, alpha_zero_version=1)

    # controller.simulate_tournament(
    #     no_games=25,
    #     seconds_per_move=0.01
    # )

    # controller.train_alpha_zero(
    #     name_for_saving="Connect_4",
    #     start_version=1
    # )
