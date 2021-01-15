from hyperparameters import *

import numpy as np
import torch


def prepare_nn_input(states_so_far, state_shape):
    nn_input = []
    # last eight positions
    if len(states_so_far) >= 8:
        for state in states_so_far[-8:]:
            nn_input.append(state)
    else:
        # fill with empty states (just zeros)
        for i in range(8 - len(states_so_far)):
            empty_state = np.zeros(state_shape)
            nn_input.append(empty_state)
        for state in states_so_far:
            nn_input.append(state)
    nn_input = np.array(nn_input)
    nn_input = nn_input.reshape(-1,8,*state_shape)
    return nn_input

def compute_search_probabilities(agent, game):
    search_probabilities = np.zeros(game.nr_actions)
    for child in agent.mcts.crnt_root.children:
        action = child.action_before_state
        prob = child.visit_count/child.father.visit_count
        search_probabilities[action] = prob
    return search_probabilities
