import pickle
import time
from collections import deque, namedtuple, defaultdict
import random
from functools import reduce
from itertools import repeat
from icecream import ic

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

    def reset(self):
        self.nn_inputs = deque(maxlen=BUFFER_SIZE)
        self.search_probabilities = deque(maxlen=BUFFER_SIZE)
        self.outcomes = deque(maxlen=BUFFER_SIZE)

    # gets a list of all states so far
    def add_experience(self, states_so_far, search_probabilities, outcome, crnt_player, state_shape):
        nn_input = prepare_nn_input(states_so_far, state_shape).squeeze(0)
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
        try:
            with open('./replay_buffers/{}_version_{}.pickle'.format(self.name_for_saving,version), 'rb') as handle:
                nn_inputs, outcomes, search_probabilities = pickle.load(handle)
            self.nn_inputs = nn_inputs
            self.outcomes = outcomes
            self.search_probabilities = search_probabilities
        except:
            self.reset()

    def sample(self, batchsize, indices=None):
        if indices is None:
            indices = [i for i in range(len(self.nn_inputs))]
        sample_indices = random.sample(indices, k=min(batchsize, len(indices)))
        nn_inputs = np.array([self.nn_inputs[sample_index] for sample_index in sample_indices])
        outcomes = np.array([self.outcomes[sample_index] for sample_index in sample_indices])
        search_probabilities = np.array([self.search_probabilities[sample_index] for sample_index in sample_indices])
        return nn_inputs, search_probabilities, outcomes


    # there might be board situations that give a different output due to the randomness in the self play process
    # if this is the case, the samples are aggregated and the mean is taken
    def consistency_check(self):
        search_probs_by_inputs = defaultdict(lambda:[])
        outcomes_by_inputs = defaultdict(lambda:[])
        for nn_input, search_prob, outcome in zip(self.nn_inputs, self.search_probabilities, self.outcomes):
            hashable_nn_input = tuple(nn_input.reshape(-1))
            list_search_probs = search_probs_by_inputs[hashable_nn_input]
            list_search_probs.append(search_prob)
            search_probs_by_inputs[hashable_nn_input] = list_search_probs

            list_outcomes = outcomes_by_inputs[hashable_nn_input]
            list_outcomes.append(outcome)
            outcomes_by_inputs[hashable_nn_input] = list_outcomes

        self.search_probabilities.clear()
        self.outcomes.clear()
        for nn_input in self.nn_inputs:
            hashable_nn_input = tuple(nn_input.reshape(-1))
            # if len(search_probs_by_inputs[hashable_nn_input])>1:
            #     print("\t\tinconsistency found")
            #     ic(search_probs_by_inputs[hashable_nn_input])

            self.search_probabilities.append(np.array(search_probs_by_inputs[hashable_nn_input]).mean(axis=0))

            # if (len(outcomes_by_inputs[hashable_nn_input]) > 1):
            #     print("\t\tinconsistency found")
            #     ic(outcomes_by_inputs[hashable_nn_input])

            self.outcomes.append(np.array(outcomes_by_inputs[hashable_nn_input]).mean())


