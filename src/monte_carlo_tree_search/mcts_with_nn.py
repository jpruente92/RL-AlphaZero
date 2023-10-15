import math
import time
from logging import Logger

from alpha_zero.neuralnetwork import NeuralNetwork
from constants.hyper_parameters import C, MAX_NR_STEPS_TRAINING
from game_logic.two_player_game import TwoPlayerGame
from monte_carlo_tree_search.mcts import MCTS
from monte_carlo_tree_search.node import Node


class MCTSWithNeuralNetwork(MCTS):

    def __init__(
            self,
            seconds_per_move: int,
            game: TwoPlayerGame,
            player_number: int,
            logger: Logger,
            neural_network: NeuralNetwork
    ):
        super().__init__(
            seconds_per_move=seconds_per_move,
            game=game,
            player_number=player_number,
            logger=logger
        )
        self.NEURAL_NETWORK = neural_network

        self.training = None

    def set_training_mode_on(self):
        self.LOGGER.debug("Set Training mode on")
        self.training = True

    def set_training_mode_off(self):
        self.LOGGER.debug("Set Training mode off")
        self.training = False

    def _compute_policy_part_of_score(self, node: Node):
        move_probability = self._compute_move_probability_with_network(node)
        return C * move_probability * math.sqrt(math.log(node.father.visit_count) / node.visit_count)

    def _compute_move_probability_with_network(self, node: Node) -> float:

        # todo: refactor
        nn_input = prepare_nn_input(node.father.GAME.all_board_states, node.GAME.STATE_SHAPE,
                                    node.CURRENT_PLAYER_NUMBER_BEFORE_STATE)
        nn_input = torch.from_numpy(nn_input).float().to(DEVICE)
        self.NETWORK.eval()
        with torch.no_grad():
            _, move_probabilities = self.NETWORK.forward(nn_input)
        self.NETWORK.train()
        move_probabilities = move_probabilities.detach().cpu().numpy()
        move_prob_network = move_probabilities[0, node.action_before_state]
        return move_prob_network

    # todo: refactor
    def _simulate(self, game, current_player, neural_network=None):
        nn_input = prepare_nn_input(game.all_board_states, game.STATE_SHAPE, current_player)
        nn_input = torch.from_numpy(nn_input).float().to(DEVICE)
        neural_network.eval()
        with torch.no_grad():
            winner, _ = neural_network.forward(nn_input, current_player).detach().cpu().numpy()
        neural_network.train()
        return winner


    def _best_action(
            self
    ) -> int:
        if self.training:
            return self._sample_best_action_by_probability()
        else:
            return self._find_action_with_highest_score()

    def _stop_condition_tree_update(self, start_time, step):
        return step <= MAX_NR_STEPS_TRAINING and time.time() - start_time < self.SECONDS_PER_MOVE

