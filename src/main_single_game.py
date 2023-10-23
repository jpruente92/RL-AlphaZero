from controlling.controller import Controller
from enums.opponent_type import OpponentType

if __name__ == '__main__':
    controller = Controller(seed=46532)
    controller.set_game_to_connect_4()

    # todo: muss agent das game verwalten? welche klassen müssen das game verwalten?
    # -    agent
    # - controller
    # -     tournament
    # -     gui
    # - mcts
    # -     node
    # -     agent compute action -> vllt reicht hier game state?


    # todo: refactor mcts with neural network
    # todo: refactor neural network
    # todo: refactor replay buffer -> do not store np arrays
    # todo: refactor alpha zero agent
    # todo: refactor alpha zero algorithm

    # todo: assert alpha 0 exists for a given version
    # todo: make training alpha 0 in run

    # todo: gui gets base class
    # todo: gui errors
    # todo: why does the gui have to be refreshed in a loop
    # todo: gui: highlighting does not work

    controller.play_game(opponent_type=OpponentType.MONTE_CARLO_TREE_SEARCH, alpha_zero_version=1, seconds_per_move=1)
