import time
from tkinter import *
from typing import Optional

import numpy as np

from constants.gui_constants import WIDTH_SQUARE, OFFSET_SQUARE, BACKGROUND_COLOR, FIELD_COLOR, COLOR_PLAYER_1, \
    COLOR_PLAYER_2, ACTION_FREEZE_IN_MS
from game_logic.game_state import GameState


class GuiConnectN(Tk):
    def __init__(
            self,
            game
    ):
        super(GuiConnectN, self).__init__()
        self.title("Connect {}".format(game.N))
        self.minsize(1000, 1000)

        self.GRAVITY_ON = game.GRAVITY_ON
        self.FONT_SIZE = 3 * game.N
        self.NO_ROWS = game.NO_ROWS
        self.NO_COLUMNS = game.NO_COLUMNS
        self.WIDTH_FIELD = self.NO_COLUMNS * WIDTH_SQUARE + (self.NO_COLUMNS + 1) * OFFSET_SQUARE
        self.HEIGHT_FIELD = (self.NO_ROWS + 1) * WIDTH_SQUARE + (self.NO_ROWS + 2) * OFFSET_SQUARE
        self.CANVAS = Canvas(self, width=self.WIDTH_FIELD, height=self.HEIGHT_FIELD, background=BACKGROUND_COLOR)

        self.labels = []
        self.time_last_user_action_update = time.time()
        self.next_user_action = None

        self.CANVAS.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.CANVAS.bind("<Button-1>", self._left_click)

        # fullscreen
        # self.attributes('-fullscreen', True)
        # self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))
        self._print_field()

    # region Public Methods
    def refresh_picture(self, game_state: GameState):
        self._update_picture(game_state)
        self.update()
        self.update_idletasks()

    # endregion Public Methods

    # region Private Methods

    def _update_picture(
            self,
            game_state: Optional[GameState]
    ):
        self.CANVAS.delete("all")

        self._print_field()
        self._print_stones(game_state)
        self._print_winner(game_state)

    def _print_field(self) -> None:
        for i in range(self.NO_ROWS):
            for j in range(self.NO_COLUMNS):
                x = OFFSET_SQUARE * (j + 1) + WIDTH_SQUARE * j
                y = OFFSET_SQUARE * (i + 1) + WIDTH_SQUARE * i
                self.CANVAS.create_rectangle(x, self.HEIGHT_FIELD - y, x + WIDTH_SQUARE,
                                             self.HEIGHT_FIELD - y - WIDTH_SQUARE, fill=FIELD_COLOR, outline="")

    def _print_stones(self, game_state: GameState) -> None:
        for i in range(self.NO_ROWS):
            fraction_of_field_for_oval = 0.8
            for j in range(self.NO_COLUMNS):
                x = OFFSET_SQUARE * (j + 1) + WIDTH_SQUARE * j
                y = OFFSET_SQUARE * (i + 1) + WIDTH_SQUARE * i

                offset_oval = (1.0 - fraction_of_field_for_oval) / 2 * WIDTH_SQUARE
                if game_state.board[i, j] == 0.0:
                    continue
                elif game_state.board[i, j] >= 1.0:
                    color = COLOR_PLAYER_1
                else:
                    color = COLOR_PLAYER_2
                if abs(game_state.board[i, j]) >= 2.0:
                    offset_oval = 0

                self.CANVAS.create_oval(x + offset_oval, self.HEIGHT_FIELD - y - offset_oval,
                                        x + WIDTH_SQUARE - offset_oval,
                                        self.HEIGHT_FIELD - y - WIDTH_SQUARE + offset_oval, fill=color, outline="")

    def _print_winner(self, game_state: GameState) -> None:
        if game_state.winner is not None:
            middle = np.floor(self.NO_COLUMNS / 2)
            x = OFFSET_SQUARE * (middle + 1) + WIDTH_SQUARE * middle
            y = OFFSET_SQUARE * (self.NO_ROWS + 1) + WIDTH_SQUARE * self.NO_ROWS
            if game_state.winner == 1:
                winner = 1
            else:
                winner = 2
            self.CANVAS.create_text((x + 0.5 * WIDTH_SQUARE, self.HEIGHT_FIELD - y - 0.5 * WIDTH_SQUARE),
                                    font=("Purisa", self.FONT_SIZE),
                                    text="The winner is player {}".format(winner),
                                    fill="black")
        elif len(game_state.feasible_actions) == 0:
            middle = np.floor(self.NO_COLUMNS / 2)
            x = OFFSET_SQUARE * (middle + 1) + WIDTH_SQUARE * middle
            y = OFFSET_SQUARE * (self.NO_ROWS + 1) + WIDTH_SQUARE * self.NO_ROWS
            self.CANVAS.create_text((x + 0.5 * WIDTH_SQUARE, self.HEIGHT_FIELD - y - 0.5 * WIDTH_SQUARE),
                                    font=("Purisa", self.FONT_SIZE), text="The game ended with a tie", fill="black")

    def _left_click(self, event):
        current_time = time.time()
        if (current_time - self.time_last_user_action_update) * 1000 >= ACTION_FREEZE_IN_MS:
            self.next_user_action = self._find_action_by_coordinates(event.x, event.y)
            self.time_last_user_action_update = current_time

    def _find_action_by_coordinates(self, x, y):
        col = int(x / (WIDTH_SQUARE + OFFSET_SQUARE))
        row = int((y - (WIDTH_SQUARE + OFFSET_SQUARE)) / (WIDTH_SQUARE + OFFSET_SQUARE))
        row = self.NO_ROWS - 1 - row
        if self.GRAVITY_ON:
            return col
        else:
            return int(row * self.NO_COLUMNS + col)

    # endregion Private Methods
