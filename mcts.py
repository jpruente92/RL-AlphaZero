import time

from node import Node


def board_equal(state_1, state_2):
    for row_1, row_2 in zip(state_1, state_2):
        for val_1, val_2 in zip(row_1, row_2):
            if val_1 != val_2:
                return False
    return True


# start player is always number 1
class MCTS():
    def __init__(self, scnds_per_move, game, player):
        self.scnds_per_move = scnds_per_move
        self.game = game
        self.root = None
        self.total_number_updates = 0
        self.total_number_moves = 0
        self.player = player

    # method returns the action computed by the monte carlo tree search
    def step(self):
        if self.root is None:
            self.root = Node(self.game, self.player, self.player)
        # if we played already we can use parts of the previous tree
        else:
            new_root = self.find_child_with_same_board()
            self.root = new_root
            if self.root is None:
                self.root = Node(self.game, self.player, self.player)
            else:
                self.root.game.print_board()

        start = time.time()
        while time.time() - start < self.scnds_per_move:
            crnt_node = self.root
            # select nodes til no children available or game stopped
            while len(crnt_node.children) > 0 and not crnt_node.terminated:
                crnt_node = crnt_node.select_node()
            # expand node (add child for each possible feasible action, make a simulation and backpropagate)
            crnt_node.expand()
            self.total_number_updates += 1
        # choose best action according to
        self.total_number_moves += 1
        return self.best_action()

    def best_action(self):
        best_action = 0
        best_score = -10000
        # action of child with highest average value is taken

        for child in self.root.children:
            score = child.sum_of_observed_values / child.visit_count
            if score > best_score:
                best_action = child.action
                best_score = score
        return best_action

    def find_child_with_same_board(self):
        board = self.game.board
        # we have to search in the grand children because both players made a move
        for child in self.root.children:
            for grandchild in child.children:
                if board_equal(board, grandchild.game.board):
                    return grandchild

        return None

