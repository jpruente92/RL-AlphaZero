import json
import string

import numpy as np

from gui.gui_replay_buffer import start_gui


def get_board_dimensions(replay_buffer_board: np.array):
    return replay_buffer_board.shape


def load_replay_buffer(path: string):
    with open(path, 'r') as replayFile:
        data = json.load(replayFile)
    return data


def calculate_boards_for_replay(replay_buffer):
    boards = []

    for state in replay_buffer:
        neural_network_input = state['neural_network_input']
        player1_board, player2_board, _ = neural_network_input[0]

        rows, cols = np.shape(player1_board)
        # Erstellen eines neuen Boards mit neutralen Feldern (0)
        board = np.zeros((rows, cols), dtype=int)

        for row in range(rows):
            for col in range(cols):
                if player1_board[row][col] == 1:
                    board[row][col] = -1  # Spieler 1
                elif player2_board[row][col] == 1:
                    board[row][col] = 1  # Spieler 2

        boards.append(board)

    return boards


def print_board(board):
    for row in board:
        for cell in row:
            if cell == -1:
                print("X", end=" ")
            elif cell == 1:
                print("O", end=" ")
            else:
                print(".", end=" ")
        print()


if __name__ == '__main__':
    print('loading replay...')
    data = load_replay_buffer("../replay_buffers/Connect_4_version_0.json")
    print('replay loaded successfully.')
    boards = calculate_boards_for_replay(data)

    # Starten der GUI mit den berechneten Boards
    start_gui(boards)
