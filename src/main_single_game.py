from controlling.controller import Controller
from enums.opponent_type import OpponentType

if __name__ == '__main__':
    controller = Controller()
    controller.set_game_to_connect_4()

    # todo: check why random wins
    # todo: muss agent das game verwalten?
    # todo: refactor mcts with neural network
    # todo: refactor neural network
    # todo: refactor replay buffer
    # todo: refactor alpha zero agent
    # todo: refactor alpha zero algorithm

    # todo: assert alpha 0 exists for a given version
    # todo: make training alpha 0 in run

    # todo: gui gets base class
    # todo: gui errors
    # todo: why does the gui have to be refreshed in a loop

    controller.play_game(opponent_type=OpponentType.MONTE_CARLO_TREE_SEARCH, alpha_zero_version=1)
