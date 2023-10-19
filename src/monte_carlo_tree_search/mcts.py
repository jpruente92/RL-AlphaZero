import math
import random
import time
from logging import Logger
from typing import Literal

from constants.hyper_parameters import *
from game_logic.two_player_game import TwoPlayerGame
from monte_carlo_tree_search.node import Node


class MCTS:
    def __init__(
            self,
            seconds_per_move: int,
            game: TwoPlayerGame,
            player_number: Literal[-1, 1],
            logger: Logger
    ):
        self.START_PLAYER = 1
        self.SECONDS_PER_MOVE = seconds_per_move
        self.GAME = game
        self.PLAYER_NUMBER = player_number
        self.LOGGER = logger

        self.total_number_updates = 0
        self.total_number_moves = 0

        self.current_root = None

    # region Public Methods
    def reset(self):
        self.current_root = None

    def step(self) -> int:

        self.total_number_moves += 1
        self._set_root_to_last_relevant_node()
        self._update_tree()
        return self._best_action()

    def log_tree(self) -> None:
        self.LOGGER.debug("TREE")
        original_root = self.current_root
        while original_root.FATHER is not None:
            original_root = original_root.FATHER
        list_of_nodes: list = [original_root]
        max_depth = 0
        while len(list_of_nodes) > 0:
            node = list_of_nodes[0]
            del list_of_nodes[0]
            for child in node.children:
                if child is not None:
                    list_of_nodes.insert(0, child)
                    if node.DEPTH > max_depth:
                        max_depth = node.DEPTH
            for i in range(0, node.DEPTH):
                print("  ", end="")
            self.LOGGER.debug(
                f"{node.ACTION_BEFORE_STATE} {node.CURRENT_PLAYER_NUMBER_BEFORE_STATE}"
                f" {node.sum_of_observed_values} {node.visit_count, node.DEPTH}")
        self.LOGGER.debug(max_depth)

    # endregion Public Methods

    # region Private Methods

    def _update_tree(self):
        start_time = time.time()
        step = 0
        while self._stop_condition_tree_update(start_time, step):
            step += 1
            current_node = self.current_root
            current_node = self._find_next_node_to_expand(current_node)
            self._expand(current_node)
            self.total_number_updates += 1

    def _stop_condition_tree_update(self, start_time, step):
        return time.time() - start_time < self.SECONDS_PER_MOVE

    def _find_next_node_to_expand(
            self,
            current_node: Node
    ) -> Node:
        while len(current_node.children) > 0 and not current_node.terminated:
            current_node = self._find_child_with_highest_uct_score(current_node)
        return current_node

    # region Expand

    def _expand(self, node: Node):

        if node.terminated:
            winner_value = node.GAME.winner
            self._back_propagate(node, winner_value)
            return

        for action in node.GAME.FEASIBLE_ACTIONS:
            child = self._create_child(action, node)
            node.children.append(child)
            value = self._compute_winner_value(child)
            self._back_propagate(child, value)

    def _create_child(
            self,
            action: int,
            node: Node
    ) -> Node:
        game_of_child = node.GAME.clone()
        game_of_child.step_if_feasible(
            action=action,
            player_number=-node.CURRENT_PLAYER_NUMBER_BEFORE_STATE
        )
        child = Node(
            game=game_of_child,
            player_number=node.PLAYER_NUMBER,
            current_player_number_before_state=-node.CURRENT_PLAYER_NUMBER_BEFORE_STATE,
            action_before_state=action,
            father=node,
            depth=node.DEPTH + 1
        )

        return child

    def _compute_winner_value(self, child: Node):
        winner = child.GAME.winner
        if winner is None:
            if len(child.GAME.FEASIBLE_ACTIONS) == 0:
                child.terminated = True
                winner_value = 0
                child.GAME.winner = 0
            else:
                child.terminated = False
                winner_value = self._simulate(child.GAME.clone(), -child.CURRENT_PLAYER_NUMBER_BEFORE_STATE)
        else:
            winner_value = winner
            child.terminated = True
        return winner_value

    def _simulate(self, game, current_player):
        while len(game.FEASIBLE_ACTIONS) > 0:
            game.step_if_feasible(random.choice(game.FEASIBLE_ACTIONS), current_player)
            current_player *= -1
            if game.winner is not None:
                return game.winner
        # if the algorithm arrives here, no feasible actions are available -> tie
        return 0

    def _back_propagate(self, node: Node, winner_value: Literal[-1, 0, 1]):
        while node is not None:
            node.visit_count += 1
            node.sum_of_observed_values += winner_value
            node = node.FATHER

    # endregion Expand

    # region UCT Score

    def _find_child_with_highest_uct_score(
            self,
            node: Node
    ) -> Node:
        best_node = None
        best_score = -10000
        for child in node.children:
            uct = self._uct_score(child)
            if uct > best_score:
                best_node = child
                best_score = uct
        return best_node

    def _uct_score(
            self,
            node: Node
    ) -> float:

        value_part_of_score = self._compute_value_part_of_score(node)
        policy_part_of_score = self._compute_policy_part_of_score(node)
        return value_part_of_score + policy_part_of_score

    def _compute_value_part_of_score(self, node):
        value_part_of_score = node.sum_of_observed_values / node.visit_count
        # when the current player is not the player of the mcts we have to invert the value part
        # because we want to choose the action best for the opponent
        if node.CURRENT_PLAYER_NUMBER_BEFORE_STATE != self.PLAYER_NUMBER:
            value_part_of_score *= -1
        # the value part has to be multiplied with the player because we want to maximize his winning chance
        value_part_of_score *= self.PLAYER_NUMBER
        return value_part_of_score

    def _compute_policy_part_of_score(self, node: Node):
        return C * math.sqrt(math.log(node.FATHER.visit_count) / node.visit_count)

    # endregion UCT Score

    def _best_action(
            self
    ) -> int:
        return self._find_action_with_highest_score()

    def _find_action_with_highest_score(self) -> int:
        best_action = 0
        best_score = -10000
        for child in self.current_root.children:
            # score = child.sum_of_observed_values / child.visit_count * self.PLAYER_NUMBER
            score = child.visit_count
            if score > best_score:
                best_action = child.ACTION_BEFORE_STATE
                best_score = score
        return best_action

    def _set_root_to_last_relevant_node(self):
        new_root = self._find_grand_child_with_same_board()
        if new_root is not None:
            self.current_root = new_root
        else:
            self.current_root = Node(
                self.GAME,
                player_number=self.PLAYER_NUMBER,
                current_player_number_before_state=-self.PLAYER_NUMBER,
                action_before_state=-1,
                father=None,
                depth=0
            )

    def _find_grand_child_with_same_board(self):
        if self.current_root is None:
            return None
        for child in self.current_root.children:
            for grandchild in child.children:
                if self.GAME.board_equal(grandchild.GAME.BOARD):
                    return grandchild
        return None

    # endregion Private Methods
