import copy
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class GameState:
    feasible_actions: list[int]
    winner: Literal[None, -1, 0, 1]
    board: np.array
    player_number_to_move: Literal[None, -1, 1]
    previous_game_state: any

    def clone(self):
        return GameState(
            feasible_actions=[i for i in self.feasible_actions],
            winner=self.winner,
            board=copy.deepcopy(self.board),
            player_number_to_move=self.player_number_to_move,
            previous_game_state=self.previous_game_state
        )

    def print_board(self):
        for row_index in range(len(self.board)):
            row = len(self.board) - row_index - 1
            row_string = ""
            for column in range(len(self.board[row])):
                row_string += " "
                if self.board[row, column] == 1:
                    row_string += "O"
                elif self.board[row, column] == -1:
                    row_string += "X"
                else:
                    row_string += " "
            print(row_string)
