import json
import os
from collections import deque, defaultdict
import random

import numpy as np

from alpha_zero.numpy_encoder import NumpyEncoder
from alpha_zero.replay_buffer_experience import ReplayBufferExperience
from constants.constants import REPLAY_BUFFER_DIR_PATH
from constants.hyper_parameters import *


class ReplayBuffer:
    def __init__(self, name_for_saving: str):
        self.experiences = deque(maxlen=BUFFER_SIZE)
        self.NAME_FOR_SAVING = name_for_saving

        os.makedirs(REPLAY_BUFFER_DIR_PATH, exist_ok=True)

    def __len__(self):
        return len(self.experiences)

    def reset(self) -> None:
        self.experiences = deque(maxlen=BUFFER_SIZE)

    def add_experience(self, experience: ReplayBufferExperience) -> None:
        assert experience.outcome is not None
        self.experiences.append(experience)

    def save_to_file(self, version: int) -> None:
        dict_list = [experience.to_dict() for experience in self.experiences]
        with open(os.path.join(REPLAY_BUFFER_DIR_PATH, f"{self.NAME_FOR_SAVING}_version_{version}"), 'w') as file:
            json.dump(dict_list, file, cls=NumpyEncoder)

    def load_from_file(self, version: int):
        with open(os.path.join(REPLAY_BUFFER_DIR_PATH, f"{self.NAME_FOR_SAVING}_version_{version}"), 'r') as file:
            loaded_list = json.load(file)
        self.experiences = deque(maxlen=BUFFER_SIZE)
        for item in loaded_list:
            self.add_experience(ReplayBufferExperience(
                neural_network_input=np.array(item["neural_network_input"]),
                search_probabilities=np.array(item["search_probabilities"]),
                outcome=item["outcome"])
            )

    # todo: refactor from here

    def sample(self,
               batchsize,
               indices=None):
        if indices is None:
            indices = [i for i in range(len(self.experiences))]
        sample_indices = random.sample(indices, k=min(batchsize, len(indices)))
        experiences = [self.experiences[sample_index] for sample_index in sample_indices]
        # todo: refactor rest of method -> currently not working
        nn_inputs = np.array([self.nn_inputs[sample_index] for sample_index in sample_indices])
        outcomes = np.array([self.outcomes[sample_index] for sample_index in sample_indices])
        search_probabilities = np.array([self.search_probabilities[sample_index] for sample_index in sample_indices])
        return nn_inputs, search_probabilities, outcomes

    # there might be board situations that give a different output due to the randomness in the self play process
    # if this is the case, the samples are aggregated and the mean is taken
    def consistency_check(self):
        search_probs_by_inputs = defaultdict(lambda: [])
        outcomes_by_inputs = defaultdict(lambda: [])
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
