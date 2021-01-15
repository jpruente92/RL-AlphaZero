import pickle
import time
from collections import deque, namedtuple
import random
from functools import reduce
from itertools import repeat

import numpy as np

from helper import prepare_nn_input
from hyperparameters import *


class Replay_buffer():
    def __init__(self, version, name_for_saving, seed):
        self.nn_inputs = deque(maxlen=BUFFER_SIZE)
        self.search_probabilities = deque(maxlen=BUFFER_SIZE)
        self.outcomes = deque(maxlen=BUFFER_SIZE)
        self.name_for_saving = name_for_saving


        self.random = random
        self.random.seed(seed)
        if version >= 0 and name_for_saving is not None:
            self.load_from_file(version)

    def __len__(self):
        return len(self.outcomes)

    # gets a list of all states so far
    def add_experience(self, states_so_far, search_probabilities, outcome, crnt_player, state_shape):
        nn_input = prepare_nn_input(states_so_far, state_shape).squeeze()
        nn_input*=-1
        self.nn_inputs.append(nn_input)
        # outcome has to be multiplied with the crnt player because the neural networks input is too
        self.outcomes.append(outcome*crnt_player)
        self.search_probabilities.append(search_probabilities)

    def save_to_file(self, version):
        combined_memory = [self.nn_inputs, self.outcomes, self.search_probabilities]
        with open('./replay_buffers/{}_version_{}.pickle'.format(self.name_for_saving,version), 'wb') as handle:
            pickle.dump(combined_memory, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_from_file(self, version):
        with open('./replay_buffers/{}_version_{}.pickle'.format(self.name_for_saving,version), 'rb') as handle:
            nn_inputs, outcomes, search_probabilities = pickle.load(handle)
        self.nn_inputs = nn_inputs
        self.outcomes = outcomes
        self.search_probabilities = search_probabilities

    def sample(self, batchsize, indices=None):
        if indices is None:
            indices = [i for i in range(len(self.nn_inputs))]
        sample_indices = random.sample(indices, k=min(batchsize, len(indices)))
        nn_inputs = np.array([self.nn_inputs[sample_index] for sample_index in sample_indices])
        outcomes = np.array([self.outcomes[sample_index] for sample_index in sample_indices])
        search_probabilities = np.array([self.search_probabilities[sample_index] for sample_index in sample_indices])
        return nn_inputs, search_probabilities, outcomes




