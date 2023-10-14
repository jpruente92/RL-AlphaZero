import math
import random

from constants.hyper_parameters import *


class Node:
    def __init__(
            self,
            game,
            player,
            crnt_player_before_state):

        self.visit_count = 0
        self.sum_of_observed_values = 0  # values exactly like in the game, changes according to the player are just made in the uct score
        self.game = game
        self.action_before_state = -1
        self.terminated = False
        self.children = []
        self.father = None
        self.player = player
        self.crnt_player_before_state = crnt_player_before_state
        self.depth = 0

    def select_node(self, neural_network=None):
        best_node = None
        best_score = -10000
        for child in self.children:
            uct = self.uct_score(child, neural_network)
            if uct > best_score:
                best_node = child
                best_score = uct
        return best_node

    def expand(self):

        # if the node is already terminated, it cannot be expanded
        if self.terminated:
            value = self.game.winner
            self.visit_count += 1
            self.sum_of_observed_values += value
            self.backpropagate(value)
            return

        # otherwise expand the node
        for action in self.game.FEASIBLE_ACTIONS:
            game_of_child = self.game.clone()
            game_of_child.step_if_feasible(action, -self.crnt_player_before_state)
            child = Node(game_of_child, self.player, -self.crnt_player_before_state)

            child.action_before_state = action
            child.father = self
            winner = game_of_child.winner
            if winner is None:
                # tie
                if len(game_of_child.FEASIBLE_ACTIONS) == 0:
                    child.terminated = True
                    value = 0
                    game_of_child.winner = 0
                # game is not over
                else:
                    child.terminated = False
                    # simulate til winner is found
                    value = self.simulate(game_of_child.clone(), -child.crnt_player_before_state)
            else:
                value = winner
                child.terminated = True
            child.visit_count = 1
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

    def simulate(self, game, crnt_player, neural_network=None):
        if neural_network is None:
            while len(game.FEASIBLE_ACTIONS) > 0:
                game.step_if_feasible(random.choice(game.FEASIBLE_ACTIONS), crnt_player)
                crnt_player *= -1
                if game.winner is not None:
                    return game.winner
            # if the algorithm arrives here, no feasible actions are available -> tie
            return 0
        else:
            nn_input = prepare_nn_input(game.all_board_states, game.STATE_SHAPE, crnt_player)
            nn_input = torch.from_numpy(nn_input).float().to(DEVICE)
            neural_network.eval()
            with torch.no_grad():
                winner, _ = neural_network.forward(nn_input, crnt_player).detach().cpu().numpy()
            neural_network.train()
            return winner

    def uct_score(self, child, neural_network=None):
        move_prob_network = 1
        if neural_network is not None:
            nn_input = prepare_nn_input(child.father.GAME.all_board_states, child.GAME.STATE_SHAPE,
                                        child.crnt_player_before_state)
            nn_input = torch.from_numpy(nn_input).float().to(DEVICE)
            neural_network.eval()
            with torch.no_grad():
                _, move_probabilities = neural_network.forward(nn_input)
            neural_network.train()
            move_probabilities = move_probabilities.detach().cpu().numpy()
            move_prob_network = move_probabilities[0, child.action_before_state]
        value_part_of_score = child.sum_of_observed_values / child.visit_count
        # when the current player is not the player of the mcts we have to invert the value part because we want to choose the action best for the oponent
        if child.crnt_player_before_state != self.player:
            value_part_of_score *= -1
        # the value part has to be multiplied with the player because we want to maximize his winning chance
        value_part_of_score *= self.player
        return value_part_of_score + C * move_prob_network * math.sqrt(
            math.log(child.father.visit_count, math.e) / child.visit_count)
