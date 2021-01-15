from tkinter import *
import numpy as np

# gui parameter
width_square = 50
offset_square = 5

background_color = "blue"
field_color = "red"
color_player_1 = "black"
color_player_2 = "gold"



class Gui(Tk):
    def __init__(self, game):
        self.game = game
        super(Gui, self).__init__()
        self.title("Connect {}".format(game.n))
        self.minsize(1000, 1000)
        self.fontsize = 3*self.game.n

        self.nr_rows = game.nr_rows
        self.nr_columns = game.nr_cols
        self.width_field = self.nr_columns * width_square + (self.nr_columns + 1) * offset_square
        self.height_field = (self.nr_rows + 1) * width_square + (self.nr_rows + 2) * offset_square

        self.labels = []
        self.canvas = Canvas(self, width=self.width_field, height=self.height_field, background=background_color)
        self.canvas.place(relx=0.5, rely=0.5, anchor=CENTER)



        self.canvas.bind("<Button-1>", self.leftclick)

        # fullscreen
        # self.attributes('-fullscreen', True)
        # self.bind("<Escape>", lambda event: self.attributes("-fullscreen", False))

        self.update_field(game.board)

    def update_field(self, board):

        self.canvas.delete("all")



        for i in range(self.nr_rows):
            fraction_of_field_for_oval = 0.8
            for j in range(self.nr_columns):
                x = offset_square * (j + 1) + width_square * j
                y = offset_square * (i + 1) + width_square * i
                self.canvas.create_rectangle(x, self.height_field - y, x + width_square,
                                             self.height_field - y - width_square, fill=field_color, outline="")
                offset_oval = (1.0 - fraction_of_field_for_oval) / 2 * width_square
                if (board[i, j] == 0.0):
                    continue
                elif (board[i, j] >= 1.0):
                    color = color_player_1

                elif (board[i, j] <= -1.0):
                    color = color_player_2
                if (abs(board[i, j]) >= 2.0):
                    offset_oval = 0

                self.canvas.create_oval(x + offset_oval, self.height_field - y - offset_oval,
                                        x + width_square - offset_oval,
                                        self.height_field - y - width_square + offset_oval, fill=color, outline="")
        # print winner
        if self.game.winner != None:
            middle = np.math.floor(self.nr_columns / 2)
            x = offset_square * (middle + 1) + width_square * middle
            y = offset_square * (self.nr_rows + 1) + width_square * self.nr_rows
            if self.game.winner == 1:
                winner = 1
            else:
                winner = 2
            self.canvas.create_text((x + 0.5 * width_square, self.height_field - y - 0.5 * width_square),
                                    font=("Purisa", self.fontsize), text="The winner is player {}".format(winner), fill="black")
        elif len(self.game.feasible_actions) == 0:
            middle = np.math.floor(self.nr_columns / 2)
            x = offset_square * (middle + 1) + width_square * middle
            y = offset_square * (self.nr_rows + 1) + width_square * self.nr_rows
            self.canvas.create_text((x + 0.5 * width_square, self.height_field - y - 0.5 * width_square),
                                    font=("Purisa", self.fontsize), text="The game ended with a tie", fill="black")

    def refresh_picture(self, board):
        self.update_field(board)
        self.update()
        self.update_idletasks()

    def leftclick(self, event):
        action = self.find_action_by_coordinates(event.x, event.y)
        if action in self.game.feasible_actions:
            self.game.user_action = action

    def find_action_by_coordinates(self, x, y):
        col = int(x/(width_square+offset_square))
        row = int((y-(width_square+offset_square))/(width_square+offset_square))
        row = self.game.nr_rows-1-row
        if self.game.gravity_on:
            return col
        else:
            return int(row*self.game.nr_cols+col)




