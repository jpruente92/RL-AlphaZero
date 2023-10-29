from controlling.controller import Controller
from enums.opponent_type import OpponentType

if __name__ == '__main__':
    controller = Controller()
    controller.set_game_to_connect_4()

    # todo: refactor alpha zero algorithm
        # todo: replay buffer von 1 auf 0 manuell zurücksetzen
        # todo: training
            # todo: what do the following torch things do: item, zero_grad, what is a tensor, backward,
            # todo: torch.no_grad, training, eval
            # todo: unterschiede für training und kein training
            # todo: check if type hints are correct

    # todo: refactor mcts with neural network
    # todo: refactor neural network
    # todo: refactor replay buffer -> do not store np arrays

    # todo: assert alpha 0 exists for a given version
    # todo: make training alpha 0 in run

    # todo: gui gets base class
    # todo: gui errors
    # todo: why does the gui have to be refreshed in a loop
    # todo: gui: highlighting of winning stones1

    controller.play_game(opponent_type=OpponentType.ALPHA_ZERO, alpha_zero_version=1, seconds_per_move=1)
