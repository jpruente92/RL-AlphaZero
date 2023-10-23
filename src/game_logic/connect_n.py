from logging import Logger
from typing import Literal

import numpy as np

from game_logic.two_player_game import TwoPlayerGame
from gui.gui_connect_n import GuiConnectN


class ConnectN(TwoPlayerGame):
    def __init__(
            self,
            logger: Logger,
            n: int,
            no_rows: int,
            no_columns: int,
            gravity_on: bool
    ):
        if gravity_on:
            no_actions = no_columns
        else:
            no_actions = no_rows * no_columns
        super().__init__(
            logger=logger,
            no_actions=no_actions,
            state_shape=(no_rows, no_columns),
            name_for_saving=f"Connect_{n}"
        )
        self.N = n
        self.NO_ROWS = no_rows
        self.NO_COLUMNS = no_columns
        self.GRAVITY_ON = gravity_on

    def set_gui_on(self):
        self.gui_on = True
        self.GUI = GuiConnectN(self)

    # region Private Methods

    # todo: refactor
    def _compute_winner(
            self,
            field: tuple,
            board: np.array,
            player_number_to_move: Literal[-1, 1]
    ) -> Literal[None, -1, 0, 1]:
        (row, column) = field
        # check horizontal
        nr_consecutive_right = 0
        for i in range(1, self.N):
            # not a feasible field
            if column + i > self.NO_COLUMNS - 1:
                break
            # not consecutive anymore
            if board[row, column + i] != player_number_to_move:
                break

            nr_consecutive_right += 1
        nr_consecutive_left = 0
        for i in range(1, self.N):
            # not a feasible field
            if column - i < 0:
                break
            # not consecutive anymore
            if board[row, column - i] != player_number_to_move:
                break
            nr_consecutive_left += 1
        if nr_consecutive_right + nr_consecutive_left + 1 >= self.N:
            return player_number_to_move

        # check vertical
        nr_consecutive_up = 0
        for i in range(1, self.N):
            # not a feasible field
            if row + i > self.NO_ROWS - 1:
                break
            # not consecutive anymore
            if board[row + i, column] != player_number_to_move:
                break
            nr_consecutive_up += 1

        nr_consecutive_down = 0
        for i in range(1, self.N):
            # not a feasible field
            if row - i < 0:
                break
            # not consecutive anymore
            if board[row - i, column] != player_number_to_move:
                break
            nr_consecutive_down += 1
        if nr_consecutive_up + nr_consecutive_down + 1 >= self.N:
            return player_number_to_move

        # check diagonal down left to up right
        nr_consecutive_right = 0
        for i in range(1, self.N):
            # not a feasible field
            if row + i > self.NO_ROWS - 1 or column + i > self.NO_COLUMNS - 1:
                break
            # not consecutive anymore
            if board[row + i, column + i] != player_number_to_move:
                break
            nr_consecutive_right += 1
        nr_consecutive_left = 0
        for i in range(1, self.N):
            # not a feasible field
            if row - i < 0 or column - i < 0:
                break
            # not consecutive anymore
            if board[row - i, column - i] != player_number_to_move:
                break
            nr_consecutive_left += 1
        if nr_consecutive_right + nr_consecutive_left + 1 >= self.N:
            return player_number_to_move

        # check diagonal up left to down right
        nr_consecutive_right = 0
        for i in range(1, self.N):
            # not a feasible field
            if row - i < 0 or column + i > self.NO_COLUMNS - 1:
                break
            # not consecutive anymore
            if board[row - i, column + i] != player_number_to_move:
                break
            nr_consecutive_right += 1
        nr_consecutive_left = 0
        for i in range(1, self.N):
            # not a feasible field
            if row + i > self.NO_ROWS - 1 or column - i < 0:
                break
            # not consecutive anymore
            if board[row + i, column - i] != player_number_to_move:
                break
            nr_consecutive_left += 1
        if nr_consecutive_right + nr_consecutive_left + 1 >= self.N:
            return player_number_to_move

        return None

    def _action_to_field(
            self,
            action: int,
            board: np.array
    ) -> tuple:
        """
        :param action: integer specifying the action of the player
            if gravity is on, player can choose a column and stone is automatically placed in the lowest row of it
            otherwise, player can choose a field
        :return:
        """

        if self.GRAVITY_ON:
            column = action
            row = self._get_lowest_free_row(column=column, board=board)
        else:
            column = action % self.NO_COLUMNS
            row = int(action / self.NO_COLUMNS)
        return row, column

    def _compute_feasible_actions(
            self,
            action: int,
            field: tuple,
            board: np.array
    ) -> list[int]:
        feasible_actions = []
        for feasible_action in range(self.NO_ACTIONS):
            if not self.GRAVITY_ON and action == feasible_action:
                continue
            if self.GRAVITY_ON and self._is_column_full(
                    column=feasible_action,
                    board=board
            ):
                continue
            feasible_actions.append(feasible_action)
        return feasible_actions

    def _get_lowest_free_row(
            self,
            column: int,
            board: np.array
    ) -> int:
        for row in range(self.NO_ROWS):
            if board[row, column] == 0:
                return row
        return self.NO_ROWS

    def _is_column_full(
            self,
            column: int,
            board: np.array
    ) -> bool:
        return self._get_lowest_free_row(column=column, board=board) == self.NO_ROWS
    # endregion Private Methods
