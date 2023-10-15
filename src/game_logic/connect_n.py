import copy
from typing import Literal

from game_logic.two_player_game import TwoPlayerGame


class ConnectN(TwoPlayerGame):
    def __init__(
            self,
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
            no_actions=no_actions,
            state_shape=(no_rows, no_columns),
        )
        self.N = n
        self.NO_ROWS = no_rows
        self.NO_COLUMNS = no_columns
        self.GRAVITY_ON = gravity_on

    # region Public Methods
    def clone(self):
        clone = ConnectN(
            n=self.N,
            no_rows=self.NO_ROWS,
            no_columns=self.NO_COLUMNS,
            gravity_on=self.GRAVITY_ON
        )
        clone.BOARD = copy.deepcopy(self.BOARD)
        clone.FEASIBLE_ACTIONS = copy.deepcopy(self.FEASIBLE_ACTIONS)
        clone.user_action = self.user_action
        clone.all_board_states = copy.deepcopy(self.all_board_states)
        return clone

    # endregion Public Methods

    # region Private Methods

    # todo: refactor
    def _update_winner(
            self,
            field: tuple,
            player_number: Literal[-1, 1]
    ) -> None:
        (row, column) = field
        # check horizontal
        nr_consecutive_right = 0
        for i in range(1, self.N):
            # not a feasible field
            if column + i > self.NO_COLUMNS - 1:
                break
            # not consecutive anymore
            if self.BOARD[row, column + i] != player_number:
                break

            nr_consecutive_right += 1
        nr_consecutive_left = 0
        for i in range(1, self.N):
            # not a feasible field
            if column - i < 0:
                break
            # not consecutive anymore
            if self.BOARD[row, column - i] != player_number:
                break
            nr_consecutive_left += 1
        if nr_consecutive_right + nr_consecutive_left + 1 >= self.N:
            self.winner = player_number
            # highlight winning stones
            if self.gui_on:
                self.BOARD[row, column] += player_number
                for i in range(1, nr_consecutive_right + 1):
                    self.BOARD[row, column + i] += player_number
                for i in range(1, nr_consecutive_left + 1):
                    self.BOARD[row, column - i] += player_number
            return

        # check vertical
        nr_consecutive_up = 0
        for i in range(1, self.N):
            # not a feasible field
            if row + i > self.NO_ROWS - 1:
                break
            # not consecutive anymore
            if self.BOARD[row + i, column] != player_number:
                break
            nr_consecutive_up += 1

        nr_consecutive_down = 0
        for i in range(1, self.N):
            # not a feasible field
            if row - i < 0:
                break
            # not consecutive anymore
            if self.BOARD[row - i, column] != player_number:
                break
            nr_consecutive_down += 1
        if nr_consecutive_up + nr_consecutive_down + 1 >= self.N:
            self.winner = player_number
            # highlight winning stones
            if self.gui_on:
                self.BOARD[row, column] += player_number
                for i in range(1, nr_consecutive_up + 1):
                    self.BOARD[row + i, column] += player_number
                for i in range(1, nr_consecutive_down + 1):
                    self.BOARD[row - i, column] += player_number
            return

        # check diagonal down left to up right
        nr_consecutive_right = 0
        for i in range(1, self.N):
            # not a feasible field
            if row + i > self.NO_ROWS - 1 or column + i > self.NO_COLUMNS - 1:
                break
            # not consecutive anymore
            if self.BOARD[row + i, column + i] != player_number:
                break
            nr_consecutive_right += 1
        nr_consecutive_left = 0
        for i in range(1, self.N):
            # not a feasible field
            if row - i < 0 or column - i < 0:
                break
            # not consecutive anymore
            if self.BOARD[row - i, column - i] != player_number:
                break
            nr_consecutive_left += 1
        if nr_consecutive_right + nr_consecutive_left + 1 >= self.N:
            self.winner = player_number
            # highlight winning stones
            if self.gui_on:
                self.BOARD[row, column] += player_number
                for i in range(1, nr_consecutive_right + 1):
                    self.BOARD[row + i, column + i] += player_number
                for i in range(1, nr_consecutive_left + 1):
                    self.BOARD[row - i, column - i] += player_number
            return

        # check diagonal up left to down right
        nr_consecutive_right = 0
        for i in range(1, self.N):
            # not a feasible field
            if row - i < 0 or column + i > self.NO_COLUMNS - 1:
                break
            # not consecutive anymore
            if self.BOARD[row - i, column + i] != player_number:
                break
            nr_consecutive_right += 1
        nr_consecutive_left = 0
        for i in range(1, self.N):
            # not a feasible field
            if row + i > self.NO_ROWS - 1 or column - i < 0:
                break
            # not consecutive anymore
            if self.BOARD[row + i, column - i] != player_number:
                break
            nr_consecutive_left += 1
        if nr_consecutive_right + nr_consecutive_left + 1 >= self.N:
            self.winner = player_number
            # highlight winning stones
            if self.gui_on:
                self.BOARD[row, column] += player_number
                for i in range(1, nr_consecutive_right + 1):
                    self.BOARD[row - i, column + i] += player_number
                for i in range(1, nr_consecutive_left + 1):
                    self.BOARD[row + i, column - i] += player_number
            return

    def _action_to_field(
            self,
            action: int
    ) -> tuple:
        """
        :param action: integer specifying the action of the player
            if gravity is on, player can choose a column and stone is automatically placed in the lowest row of it
            otherwise, player can choose a field
        :return:
        """

        if self.GRAVITY_ON:
            column = action
            row = self._get_lowest_free_row(column)
        else:
            column = action % self.NO_COLUMNS
            row = int(action / self.NO_COLUMNS)
        return row, column

    def _update_feasible_actions(
            self,
            action: int,
            field: tuple
    ) -> None:
        if self.GRAVITY_ON:
            if self._is_column_full(field[1]):
                self.FEASIBLE_ACTIONS = [x for x in self.FEASIBLE_ACTIONS if x != action]
        else:
            self.FEASIBLE_ACTIONS = [x for x in self.FEASIBLE_ACTIONS if x != action]

    def _get_lowest_free_row(
            self,
            column: int
    ) -> int:
        for row in range(self.NO_ROWS):
            if self.BOARD[row, column] == 0:
                return row
        return self.NO_ROWS

    def _is_column_full(self, column):
        return self._get_lowest_free_row(column) == self.NO_ROWS
    # endregion Private Methods
