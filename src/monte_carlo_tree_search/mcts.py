import math
import random
import time
from logging import Logger
from typing import Literal, Optional

import numpy as np

from constants.hyper_parameters import *
from game_logic.game_state import GameState
from game_logic.two_player_game import TwoPlayerGame
from monte_carlo_tree_search.node import Node


class MCTS:
    def __init__(
            self,
            game: TwoPlayerGame,
            seconds_per_move: int,
            player_number: Literal[-1, 1],
            logger: Logger
    ):
        self.GAME = game
        self.START_PLAYER = 1
        self.SECONDS_PER_MOVE = seconds_per_move
        self.player_number = player_number
        self.LOGGER = logger

        self.total_number_updates = 0
        self.total_number_moves = 0

        self.current_root: Optional[Node] = None

    # region Public Methods
    def reset(self):
        self.current_root = None

    def step(
            self,
            game_state: GameState
    ) -> int:

        self.total_number_moves += 1
        self._set_root_to_last_relevant_node(
            game_state=game_state
        )
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

    def _update_tree(
            self
    ):
        start_time = time.time()
        step = 0
        while self._stop_condition_tree_update(start_time, step):
            step += 1
            current_node = self._find_next_node_to_expand(self.current_root)
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

    def _expand(
            self,
            node: Node
    ) -> None:

        if node.terminated:
            winner_value = node.GAME_STATE.winner
            self._back_propagate(node, winner_value)
            return

        for action in node.GAME_STATE.feasible_actions:
            child = self._create_child(action, node)
            node.children.append(child)
            value = self._compute_winner_value(child=child)
            self._back_propagate(child, value)
            self.LOGGER.debug(f"Added new child with depth: {child.DEPTH}")

    def _create_child(
            self,
            action: int,
            node: Node
    ) -> Node:

        game_state = self.GAME.step_if_feasible(
            action=action,
            game_state=node.GAME_STATE
        )
        child = Node(
            game_state=game_state,
            action_before_state=action,
            father=node,
            depth=node.DEPTH + 1
        )

        return child

    def _compute_winner_value(
            self,
            child: Node,
    ) -> Literal[-1, 0, 1]:
        winner = child.GAME_STATE.winner
        if winner is None:
            if len(child.GAME_STATE.feasible_actions) == 0:
                child.terminated = True
                winner_value = 0
                child.GAME_STATE.winner = 0
            else:
                child.terminated = False
                winner_value = self._compute_winner_by_simulation(
                    game_state=child.GAME_STATE
                )
        else:
            winner_value = winner
            child.terminated = True
        return winner_value

    def _compute_winner_by_simulation(
            self,
            game_state: GameState,
    ) -> Literal[-1, 0, 1]:
        winner = 0
        current_game_state = game_state
        while len(current_game_state.feasible_actions) > 0:
            action = random.choice(current_game_state.feasible_actions)
            current_game_state = self.GAME.step_if_feasible(
                action=action,
                game_state=current_game_state
            )
            if current_game_state.winner is not None:
                winner = current_game_state.winner
                break

        return winner

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
        if node.GAME_STATE.player_number_to_move != self.player_number:
            value_part_of_score *= -1
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

    def _set_root_to_last_relevant_node(
            self,
            game_state: GameState,
    ) -> None:
        new_root = self._find_grand_child_with_same_board(game_state)
        if new_root is not None:
            self.current_root = new_root
        else:
            self.current_root = Node(
                game_state=game_state.clone(),
                action_before_state=-1,
                father=None,
                depth=0
            )

    def _find_grand_child_with_same_board(
            self,
            game_state: GameState
    ):

        if self.current_root is None:
            self.LOGGER.warning("Could not find child with same board!")
            return None
        for child in self.current_root.children:
            for grandchild in child.children:
                if self._board_equal(grandchild.GAME_STATE.board, game_state.board):
                    return grandchild
        self.LOGGER.warning("Could not find child with same board!")
        return None

    def _board_equal(
            self,
            board_1: np.array,
            board_2: np.array
    ) -> bool:
        return np.array_equal(board_1, board_2)

    # endregion Private Methods
