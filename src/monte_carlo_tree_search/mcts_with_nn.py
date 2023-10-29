import math
import random
import time
from logging import Logger
from typing import Literal

from constants.hyper_parameters import C, MAX_NR_STEPS_TRAINING
from game_logic.game_state import GameState
from game_logic.two_player_game import TwoPlayerGame
from monte_carlo_tree_search.mcts import MCTS
from monte_carlo_tree_search.node import Node
from neural_network.network_manager import NetworkManagerBase


class MCTSWithNeuralNetwork(MCTS):

    def __init__(
            self,
            seconds_per_move: int,
            game: TwoPlayerGame,
            player_number: Literal[-1, 1],
            logger: Logger,
            network_manager: NetworkManagerBase
    ):
        super().__init__(
            seconds_per_move=seconds_per_move,
            game=game,
            player_number=player_number,
            logger=logger
        )
        self.NETWORK_MANAGER = network_manager

        self.training = None

    def clone(
            self,
            network_manager: NetworkManagerBase
    ):
        clone = MCTSWithNeuralNetwork(
            seconds_per_move=self.SECONDS_PER_MOVE,
            game=self.GAME,
            player_number=self.player_number,
            logger=self.LOGGER,
            network_manager=network_manager
        )
        return clone

    def set_training_mode_on(self):
        self.LOGGER.debug("Set Training mode on")
        self.training = True

    def set_training_mode_off(self):
        self.LOGGER.debug("Set Training mode off")
        self.training = False

    def _compute_policy_part_of_score(self, node: Node):
        move_probability = self._compute_move_probability_with_network(node) if self.NETWORK_MANAGER.VERSION > 0 else 1
        return C * move_probability * math.sqrt(math.log(node.FATHER.visit_count) / node.visit_count)

    def _compute_move_probability_with_network(self, node: Node) -> float:
        # todo: check if the input is correct to get the move probabilities
        _, move_probabilities = self.NETWORK_MANAGER.evaluate(
            states_so_far=node.FATHER.GAME_STATE.get_list_of_all_board_states_from_start_to_end(),
            current_player=node.FATHER.GAME_STATE.player_number_to_move
        )
        move_prob_network = move_probabilities[0, node.ACTION_BEFORE_STATE]
        return move_prob_network

    def _compute_winner_by_simulation(
            self,
            game_state: GameState,
    ) -> Literal[-1, 0, 1]:
        if self.NETWORK_MANAGER.VERSION < 1:
            return super()._compute_winner_by_simulation(game_state)

        winner, _ = self.NETWORK_MANAGER.evaluate(
            states_so_far=game_state.get_list_of_all_board_states_from_start_to_end(),
            current_player=game_state.player_number_to_move
        )
        return winner

    def _sample_best_action_by_probability(self) -> int:
        actions = []
        probabilities = []
        for child in self.current_root.children:
            actions.append(child.ACTION_BEFORE_STATE)
            probabilities.append(child.visit_count / self.current_root.visit_count)
        return random.choices(actions, weights=probabilities, k=1)[0]

    def _best_action(
            self
    ) -> int:
        if self.training:
            return self._sample_best_action_by_probability()
        else:
            return self._find_action_with_highest_score()

    def _stop_condition_tree_update(self, start_time, step):
        return step <= MAX_NR_STEPS_TRAINING and time.time() - start_time < self.SECONDS_PER_MOVE
