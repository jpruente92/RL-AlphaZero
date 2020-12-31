import copy
import time

import numpy as np

from gui import Gui


class Game():
    def __init__(self,random, n, nr_rows, nr_cols, gravity_on, show_output=False, sleeping_time=1):
        self.random = random
        self.n = n
        self.nr_rows = nr_rows
        self.nr_cols = nr_cols
        self.gravity_on = gravity_on

        self.sleeping_time = sleeping_time

        if gravity_on:
            self.nr_actions = nr_cols
        else:
            self.nr_actions = nr_rows * nr_cols
        self.state_shape = (nr_rows, nr_cols)
        self.board = np.zeros(self.state_shape)
        self.winner = None
        self.feasible_actions = [i for i in range(self.nr_actions)]
        self.user_action = None

        self.show_output = show_output
        if show_output:
            self.gui = Gui(self)
            self.gui.refresh_picture(self.board)

    def clone(self):
        clone = Game(self.random, self.n, self.nr_rows, self.nr_cols, self.gravity_on, show_output=False,sleeping_time=0)
        clone.board = copy.deepcopy(self.board)
        clone.feasible_actions = copy.deepcopy(self.feasible_actions)
        clone.user_action = self.user_action
        return clone

    def reset(self):
        self.board = np.zeros(*self.state_shape)
        self.winner = None
        self.feasible_actions = [i for i in range(self.nr_actions)]
        self.user_action = None

    def step_if_feasible(self, action, player):
        if self.gravity_on:
            # try to place stone in column as low as possible
            for row in range(self.nr_rows):
                # last stone in column is placed, action for this column not feasible anymore
                if row == self.nr_rows - 1:
                    self.feasible_actions = [x for x in self.feasible_actions if x != action]
                if self.board[row, action] == 0:
                    self.board[row, action] = player
                    self.check_if_won(row, action, player)
                    break

        else:
            row, col = self.translate_action(action)
            if self.board[row, col] == 0:
                self.board[row, col] = player
                self.check_if_won(row, col, player)
            # action used and therefore not feasible anymore
            self.feasible_actions = [x for x in self.feasible_actions if x != action]
        if self.show_output:
            self.gui.refresh_picture(self.board)

    def check_if_won(self, row, column, player):
        # check horizontal
        nr_consecutive_right = 0
        for i in range(1, self.n):
            # not a feasible field
            if column + i > self.nr_cols - 1:
                break
            # not consecutive anymore
            if self.board[row, column + i] != player:
                break

            nr_consecutive_right += 1
        nr_consecutive_left = 0
        for i in range(1, self.n):
            # not a feasible field
            if column - i < 0:
                break
            # not consecutive anymore
            if self.board[row, column - i] != player:
                break
            nr_consecutive_left += 1
        if nr_consecutive_right + nr_consecutive_left + 1 >= self.n:
            self.winner = player
            # highlight winning stones
            if self.show_output:
                self.board[row, column] += player
                for i in range(1, nr_consecutive_right + 1):
                    self.board[row, column + i] += player
                for i in range(1, nr_consecutive_left + 1):
                    self.board[row, column - i] += player
            return

        # check vertical
        nr_consecutive_up = 0
        for i in range(1, self.n):
            # not a feasible field
            if row + i > self.nr_rows - 1:
                break
            # not consecutive anymore
            if self.board[row + i, column] != player:
                break
            nr_consecutive_up += 1

        nr_consecutive_down = 0
        for i in range(1, self.n):
            # not a feasible field
            if row - i < 0:
                break
            # not consecutive anymore
            if self.board[row - i, column] != player:
                break
            nr_consecutive_down += 1
        if nr_consecutive_up + nr_consecutive_down + 1 >= self.n:
            self.winner = player
            # highlight winning stones
            if self.show_output:
                self.board[row, column] += player
                for i in range(1, nr_consecutive_up + 1):
                    self.board[row + i, column] += player
                for i in range(1, nr_consecutive_down + 1):
                    self.board[row - i, column] += player
            return

        # check diagonal down left to up right
        nr_consecutive_right = 0
        for i in range(1, self.n):
            # not a feasible field
            if row + i > self.nr_rows - 1 or column + i > self.nr_cols - 1:
                break
            # not consecutive anymore
            if self.board[row + i, column + i] != player:
                break
            nr_consecutive_right += 1
        nr_consecutive_left = 0
        for i in range(1, self.n):
            # not a feasible field
            if row - i < 0 or column - i < 0:
                break
            # not consecutive anymore
            if self.board[row - i, column - i] != player:
                break
            nr_consecutive_left += 1
        if nr_consecutive_right + nr_consecutive_left + 1 >= self.n:
            self.winner = player
            # highlight winning stones
            if self.show_output:
                self.board[row, column] += player
                for i in range(1, nr_consecutive_right + 1):
                    self.board[row + i, column + i] += player
                for i in range(1, nr_consecutive_left + 1):
                    self.board[row - i, column - i] += player
            return

        # check diagonal up left to down right
        nr_consecutive_right = 0
        for i in range(1, self.n):
            # not a feasible field
            if row - i < 0 or column + i > self.nr_cols - 1:
                break
            # not consecutive anymore
            if self.board[row - i, column + i] != player:
                break
            nr_consecutive_right += 1
        nr_consecutive_left = 0
        for i in range(1, self.n):
            # not a feasible field
            if row + i > self.nr_rows - 1 or column - i < 0:
                break
            # not consecutive anymore
            if self.board[row + i, column - i] != player:
                break
            nr_consecutive_left += 1
        if nr_consecutive_right + nr_consecutive_left + 1 >= self.n:
            self.winner = player
            # highlight winning stones
            if self.show_output:
                self.board[row, column] += player
                for i in range(1, nr_consecutive_right + 1):
                    self.board[row - i, column + i] += player
                for i in range(1, nr_consecutive_left + 1):
                    self.board[row + i, column - i] += player
            return

    def translate_action(self, action):
        return int(action / self.nr_cols), action % self.nr_cols

    def print_board(self):
        for row in range(0, self.nr_rows):
            for col in range(0, self.nr_cols):
                print('{:2d}'.format(int(self.board[self.nr_rows - 1 - row, col])), end=" ")

            print()

    def start_game(self, player_1, player_2):
        players = [player_1, player_2]
        crnt_player = players[self.random.randint(0, 1)]
        while self.winner is None and len(self.feasible_actions) > 0:
            action = crnt_player.compute_action(self)
            self.step_if_feasible(action, crnt_player.player)
            time.sleep(self.sleeping_time)
            crnt_player = player_1 if crnt_player is player_2 else player_2
        while True:
            self.gui.refresh_picture(self.board)
