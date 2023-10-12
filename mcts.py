import time

from node import Node
from hyperparameters import *
import random


def board_equal(state_1, state_2):
    for row_1, row_2 in zip(state_1, state_2):
        for val_1, val_2 in zip(row_1, row_2):
            if val_1 != val_2:
                return False
    return True


# start player is always number 1
class MCTS():
    def __init__(self, scnds_per_move, game, player, seed, network=None):
        self.scnds_per_move = scnds_per_move
        self.game = game
        self.crnt_root = None
        self.total_number_updates = 0
        self.total_number_moves = 0
        self.player = player
        self.network = network
        random.seed(seed)

    def reset(self):
        self.crnt_root = None

    # method returns the action computed by the monte carlo tree search
    def step(self, training):
        if self.crnt_root is None:
            self.crnt_root = Node(self.game, player=self.player, crnt_player_before_state=-self.player)
        # if we played already we can use parts of the previous tree
        else:
            new_root = self.find_child_with_same_board()
            self.crnt_root = new_root
            if self.crnt_root is None:
                self.crnt_root = Node(self.game, player=self.player, crnt_player_before_state=-self.player)

        start = time.time()
        step = 1
        while True:
            if step > MAX_NR_STEPS_TRAINING:
                break
            step +=1
            if time.time() - start > self.scnds_per_move:
                break
            crnt_node = self.crnt_root
            # select nodes til no children available or game stopped
            while len(crnt_node.children) > 0 and not crnt_node.terminated:
                crnt_node = crnt_node.select_node(self.network)
            # expand node (add child for each possible feasible action, make a simulation and backpropagate)
            crnt_node.expand()
            self.total_number_updates += 1
        # choose best action according to highest average value
        self.total_number_moves += 1
        return self.best_action(training)

    def best_action(self, training):
        best_action = 0
        best_score = -10000
        if training and self.network is not None:
            # sample value by search probabilities
            actions = []
            probabilities = []
            for child in self.crnt_root.children:
                actions.append(child.action_before_state)
                probabilities.append(child.visit_count / self.crnt_root.visit_count)
            return random.choices(actions, weights=probabilities,k=1)[0]
        else:
            # if competitive or no neural network
            # action of child with highest average value is taken
            for child in self.crnt_root.children:
                # score = child.sum_of_observed_values / child.visit_count
                # multiply score with the mcts player because player -1 favors scores of -1
                # score *= self.player
                score = child.visit_count
                # print(score, child.visit_count, child.sum_of_observed_values, child.action_before_state)
                # print(child.action_before_state, child.sum_of_observed_values, child.visit_count, score)
                if score > best_score:
                    best_action = child.action_before_state
                    best_score = score
            # print()
            # for child in best_child.children:
            #     score = child.sum_of_observed_values / child.visit_count
                # print("\t",score, child.visit_count, child.sum_of_observed_values, child.action_before_state)
            # print()
            return best_action

    def find_child_with_same_board(self):
        board = self.game.board
        # we have to search in the grand children because both players made a move
        for child in self.crnt_root.children:
            for grandchild in child.children:
                if board_equal(board, grandchild.game.board):
                    return grandchild
        return None

    def print_tree(self):
        print("TREE")
        # find original_root
        original_root=self.crnt_root
        while original_root.father is not None:
            original_root = original_root.father
        list_of_nodes = [original_root]
        max_depth = 0
        while len(list_of_nodes)>0:
            node = list_of_nodes[0]
            del list_of_nodes[0]
            for child in node.children :
                if child is not None:
                    list_of_nodes.insert(0,child)
                    if(node.depth>max_depth):
                        max_depth=node.depth
            for i in range (0,node.depth):
                print("  ",end="")
            print(node.action_before_state, node.crnt_player_before_state, node.sum_of_observed_values, node.visit_count, node.depth)
        print(max_depth)