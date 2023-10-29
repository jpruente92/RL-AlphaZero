from typing import Literal

import numpy as np


class GameState:
    def __init__(
            self,
            feasible_actions: list[int],
            winner: Literal[None, -1, 0, 1],
            board: np.array,
            player_number_to_move: Literal[None, -1, 1],
            previous_game_state: any,
    ):
        assert player_number_to_move != 0
        self.feasible_actions = feasible_actions
        self.winner = winner
        self.board = board
        self.player_number_to_move = player_number_to_move
        self.previous_game_state = previous_game_state

        self.all_board_states = None

    def __str__(self):
        return f"feasible actions:\n\t\t {self.feasible_actions}\n" \
               f"player number to move:\n\t\t {self.player_number_to_move}\n" \
               f"board:\n{self._pretty_print_board()}\n" \
               f"winner:\n\t\t {self.winner}"

    def get_number_moves_made(self):
        return np.count_nonzero(self.board)

    def print_board(self):
        print(self._pretty_print_board())

    def get_list_of_all_board_states_from_start_to_end(self) -> list[np.array]:
        if self.all_board_states is None:
            self._set_all_board_states()
        return self.all_board_states

    def _set_all_board_states(self):
        self.all_board_states = []
        current_game_state = self
        while True:
            self.all_board_states.append(current_game_state.board)
            if current_game_state.previous_game_state is None:
                break
            current_game_state = current_game_state.previous_game_state
        self.all_board_states = self.all_board_states[::-1]

    def _pretty_print_board(self) -> str:
        output = ""
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
            output += row_string + "\n"
        return output
