import tkinter as tk

import numpy as np

boards = []
current_turn = 0
cell_size = 50
rows = 0
cols = 0


def draw_board(canvas, board):
    canvas.delete("all")  # Bestehendes Zeichnen löschen
    cell_size = 50

    for y in range(rows):
        for x in range(cols):
            x1 = x * cell_size
            y1 = (rows - 1 - y) * cell_size  # Y-Achse gespiegelt
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            canvas.create_rectangle(x1, y1, x2, y2, fill='white', outline='black')
            if board[y, x] == 1:
                canvas.create_oval(x1, y1, x2, y2, fill='blue', outline='blue')
            elif board[y, x] == -1:
                canvas.create_oval(x1, y1, x2, y2, fill='red', outline='red')


def on_next():
    global current_turn
    current_turn = min(current_turn + 1, len(boards) - 1)
    draw_board(canvas, boards[current_turn])


def on_prev():
    global current_turn
    current_turn = max(current_turn - 1, 0)
    draw_board(canvas, boards[current_turn])


def keypress(event):
    global current_turn
    if event.keysym == 'Right':
        on_next()
    elif event.keysym == 'Left':
        on_prev()


def start_gui(boards_input: np.array):
    global canvas, current_turn, boards, rows, cols

    # Canvas erzeugen
    root = tk.Tk()
    root.title("Connect Four")

    canvas = tk.Canvas(root)
    rows, cols = boards_input[0].shape
    canvas_width = cols * cell_size
    canvas_height = rows * cell_size
    canvas.config(width=canvas_width, height=canvas_height)
    canvas.grid(row=1, column=0, columnspan=2)

    # Buttons für Züge
    next_button = tk.Button(root, text="Nächster Zug", command=on_next)
    next_button.grid(row=0, column=1)
    prev_button = tk.Button(root, text="Vorheriger Zug", command=on_prev)
    prev_button.grid(row=0, column=0)

    # Tastatureingabe ermöglichen
    root.bind('<Left>', keypress)
    root.bind('<Right>', keypress)

    # Initialisiere die Spielbretter
    current_turn = 0
    boards = boards_input
    draw_board(canvas, boards[current_turn])
    root.mainloop()
