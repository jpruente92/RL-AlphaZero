from typing import Literal

from game_logic.two_player_game import TwoPlayerGame


class Node:
    def __init__(
            self,
            game: TwoPlayerGame,
            player_number: int,
            current_player_number_before_state: Literal[-1, 1],
            action_before_state: int,
            father,
            depth: int
    ):
        self.GAME = game
        self.CURRENT_PLAYER_NUMBER_BEFORE_STATE = current_player_number_before_state
        self.PLAYER_NUMBER = player_number
        self.ACTION_BEFORE_STATE = action_before_state
        self.FATHER = father
        self.DEPTH = depth

        self.terminated = False
        self.children = []
        self.visit_count = 0
        self.sum_of_observed_values = 0
