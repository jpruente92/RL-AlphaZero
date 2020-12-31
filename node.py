import math
from functools import reduce

from hyperparameters import *


def uct_score(child):
    return child.father.crnt_player * child.sum_of_observed_values / child.visit_count + \
           C * math.sqrt(math.log(child.father.visit_count, math.e) / child.visit_count)


class Node():
    def __init__(self, game, player, crnt_player):
        self.visit_count = 0
        self.sum_of_observed_values = 0  # values for player
        self.game = game
        self.action = -1
        self.terminated = False
        self.children = []
        self.father = None
        self.player = player
        self.crnt_player = crnt_player
        self.depth = 0

    def select_node(self):
        best_node = None
        best_score = -10000
        for child in self.children:
            uct = uct_score(child)
            sum = 0
            for row in child.game.board:
                sum += reduce(lambda a, b: abs(a) + abs(b), row)


            if uct > best_score:
                best_node = child
                best_score = uct
        return best_node

    def expand(self):

        # if the node is already terminated, it cannot be expanded
        if self.terminated:
            value = self.game.winner * self.player
            self.visit_count += 1
            self.sum_of_observed_values += value
            self.depth = self.depth + 1
            self.backpropagate(value)
            return

        # otherwise expand the node
        for action in self.game.feasible_actions:
            game_of_child = self.game.clone()
            # print("\t\t\t",len(self.game.feasible_actions), len(game_of_child.feasible_actions))
            game_of_child.step_if_feasible(action, self.crnt_player)
            # print("\t\t\t", len(self.game.feasible_actions), len(game_of_child.feasible_actions))
            child = Node(game_of_child, self.player, -self.crnt_player)

            child.action = action
            child.father = self
            winner = game_of_child.winner
            if winner is None:
                # tie
                if len(game_of_child.feasible_actions) == 0:
                    child.terminated = True
                    winner = 0
                    game_of_child.winner = 0
                # game is not over
                else:
                    child.terminated = False
                    # simulate til winner is found
                    winner = simulate(game_of_child.clone(), child.crnt_player)
            else:
                child.terminated = True
            child.visit_count = 1
            # one point if won, 0 points for tie and -1 point for loss times the player
            value = winner * self.player
            child.sum_of_observed_values = value
            child.depth = self.depth + 1
            self.children.append(child)
            child.backpropagate(value)

    def backpropagate(self, value):
        crnt_node = self
        while crnt_node.father is not None:
            crnt_node = crnt_node.father
            crnt_node.visit_count += 1
            crnt_node.sum_of_observed_values += value
        # print("backpropagate",self.depth, crnt_node.depth)


def simulate(game, player):
    crnt_player = player
    while len(game.feasible_actions) > 0:
        game.step_if_feasible(game.random.choice(game.feasible_actions), crnt_player)
        crnt_player *= -1
        if game.winner is not None:
            return game.winner
    # if the algorithm arrives here, no feasible actions are available -> tie
    return 0
