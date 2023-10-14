import time
from logging import Logger

from alpha_zero.neuralnetwork import NeuralNetwork
from game_logic.two_player_game import TwoPlayerGame
from constants.hyper_parameters import *
import random

from monte_carlo_tree_search.node import Node


class MCTS:
    def __init__(
            self,
            seconds_per_move: int,
            game: TwoPlayerGame,
            player_number: int):
        self.START_PLAYER = 1
        self.SECONDS_PER_MOVE = seconds_per_move
        self.GAME = game
        self.PLAYER_NUMBER = player_number

        self.total_number_updates = 0
        self.total_number_moves = 0

        self.NETWORK = None
        self.current_root = None

    # region Public Methods
    def set_network(
            self,
            network: NeuralNetwork):
        self.NETWORK = network

    def reset(self):
        self.current_root = None

    def step(
            self,
            training: bool
    ) -> int:
        self.total_number_moves += 1
        self._set_root_to_last_relevant_node()
        self._update_tree()
        return self._best_action(training)

    def log_tree(self, logger: Logger) -> None:
        logger.debug("TREE")
        original_root = self.current_root
        while original_root.father is not None:
            original_root = original_root.father
        list_of_nodes: list = [original_root]
        max_depth = 0
        while len(list_of_nodes) > 0:
            node = list_of_nodes[0]
            del list_of_nodes[0]
            for child in node.children:
                if child is not None:
                    list_of_nodes.insert(0, child)
                    if node.depth > max_depth:
                        max_depth = node.depth
            for i in range(0, node.depth):
                print("  ", end="")
            logger.debug(
                f"{node.action_before_state} {node.crnt_player_before_state}"
                f" {node.sum_of_observed_values} {node.visit_count, node.depth}")
        logger.debug(max_depth)

    # endregion Public Methods

    # region Private Methods

    def _update_tree(self):
        start_time = time.time()
        step = 0
        while step <= MAX_NR_STEPS_TRAINING and time.time() - start_time < self.SECONDS_PER_MOVE:
            step += 1
            current_node = self.current_root
            current_node = self._find_next_node_to_expand(current_node)
            current_node.expand()
            self.total_number_updates += 1

    def _find_next_node_to_expand(
            self,
            current_node: Node
    ) -> Node:
        while len(current_node.children) > 0 and not current_node.terminated:
            current_node = current_node.select_node(self.NETWORK)
        return current_node

    def _best_action(
            self,
            training: bool
    ) -> int:
        if training:
            return self._sample_best_action_by_probability()
        else:
            return self._find_action_with_highest_score()

    def _find_action_with_highest_score(self) -> int:
        best_action = 0
        best_score = -10000
        for child in self.current_root.children:
            # score = child.sum_of_observed_values / child.visit_count * self.PLAYER_NUMBER
            score = child.visit_count
            if score > best_score:
                best_action = child.action_before_state
                best_score = score
        return best_action

    def _sample_best_action_by_probability(self) -> int:
        actions = []
        probabilities = []
        for child in self.current_root.children:
            actions.append(child.action_before_state)
            probabilities.append(child.visit_count / self.current_root.visit_count)
        return random.choices(actions, weights=probabilities, k=1)[0]

    def _set_root_to_last_relevant_node(self):
        new_root = self._find_grand_child_with_same_board()
        if new_root is not None:
            self.current_root = new_root
        else:
            self.current_root = Node(
                self.GAME,
                player=self.PLAYER_NUMBER,
                crnt_player_before_state=-self.PLAYER_NUMBER
            )

    def _find_grand_child_with_same_board(self):
        if self.current_root is None:
            return None
        for child in self.current_root.children:
            for grandchild in child.children:
                if self.GAME.board_equal(grandchild.game.BOARD):
                    return grandchild
        return None

    # endregion Private Methods
