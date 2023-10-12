from copy import deepcopy

from hyperparameters import *

import numpy as np
import torch
from icecream import ic

def prepare_nn_input(states_so_far, state_shape, current_player):
    nn_input = []
    # last NR_BOARD_STATES_SAVED positions
    if len(states_so_far) >= NR_BOARD_STATES_SAVED:
        for state in states_so_far[-NR_BOARD_STATES_SAVED:]:
            nn_input.append(state_to_3dim_state(state,current_player))
    else:
        # fill with empty states (just zeros)
        for i in range(NR_BOARD_STATES_SAVED - len(states_so_far)):
            empty_state = np.zeros(state_shape)
            nn_input.append(state_to_3dim_state(empty_state, current_player))
        for state in states_so_far:
            nn_input.append(state_to_3dim_state(state,current_player))
    nn_input = np.array(nn_input)
    nn_input = nn_input.reshape(-1,3*NR_BOARD_STATES_SAVED,*state_shape)
    return nn_input

def state_to_3dim_state(state, current_player):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range(len(state)):
        layer_1.append([])
        layer_2.append([])
        layer_3.append([])
        for j in range(len(state[i])):
            if state[i][j]==1:
                layer_1[i].append(1)
                layer_2[i].append(0)
            if state[i][j]==-1:
                layer_1[i].append(0)
                layer_2[i].append(1)
            if state[i][j]==0:
                layer_1[i].append(0)
                layer_2[i].append(0)
            if current_player==1:
                layer_3[i].append(1)
            else:
                layer_3[i].append(0)
    return [layer_1, layer_2, layer_3]



def compute_search_probabilities(agent, game):
    search_probabilities = np.zeros(game.nr_actions)
    for child in agent.mcts.crnt_root.children:
        action = child.action_before_state
        prob = child.visit_count/child.father.visit_count
        search_probabilities[action] = prob
    return search_probabilities
