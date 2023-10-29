import copy
import json
import os
from collections import deque, defaultdict
from logging import Logger
from statistics import mean

import numpy as np

from alpha_zero.numpy_encoder import NumpyEncoder
from alpha_zero.replay_buffer_experience import ReplayBufferExperience
from constants.constants import REPLAY_BUFFER_DIR_PATH
from constants.hyper_parameters import *


class ReplayBuffer:
    def __init__(
            self,
            logger: Logger,
            name_for_saving: str,
            version: int
    ):
        self.LOGGER = logger
        self.NAME_FOR_SAVING = name_for_saving
        self.VERSION = version

        self.experiences: deque[ReplayBufferExperience] = deque(maxlen=BUFFER_SIZE)

        os.makedirs(REPLAY_BUFFER_DIR_PATH, exist_ok=True)
        self._load_from_file()

    def __len__(self):
        return len(self.experiences)

    def clone(self):
        clone = ReplayBuffer(
            logger=self.LOGGER,
            name_for_saving=self.NAME_FOR_SAVING,
            version=self.VERSION
        )
        clone.experiences = copy.deepcopy(self.experiences)
        return clone

    def reset(self) -> None:
        self.experiences = deque(maxlen=BUFFER_SIZE)

    def add_experience(self, experience: ReplayBufferExperience) -> None:
        assert experience.outcome is not None
        self.experiences.append(experience)

    def add_experiences(self, experiences: deque[ReplayBufferExperience]) -> None:
        self.experiences.extend(experiences)

    def save_to_file(self, version: int) -> None:
        self._make_experiences_consistent()
        dict_list = [experience.to_dict() for experience in self.experiences]
        file_path = os.path.join(REPLAY_BUFFER_DIR_PATH, f"{self.NAME_FOR_SAVING}_version_{version}.json")
        with open(file_path, 'w') as file:
            json.dump(dict_list, file, cls=NumpyEncoder)
        self.LOGGER.info(f"Stored replay buffer to {file_path}")

    def _load_from_file(self):
        file_path = os.path.join(REPLAY_BUFFER_DIR_PATH, f"{self.NAME_FOR_SAVING}_version_{self.VERSION}.json")
        if self.VERSION >= 0 and self.NAME_FOR_SAVING is not None and os.path.exists(file_path):
            self.LOGGER.info(f"Loaded replay buffer from {file_path}")
            with open(file_path, 'r') as file:
                loaded_list = json.load(file)
            for item in loaded_list:
                self.add_experience(ReplayBufferExperience(
                    neural_network_input=np.array(item["neural_network_input"]),
                    search_probabilities=np.array(item["search_probabilities"]),
                    outcome=item["outcome"])
                )

    # there might be board situations that give a different output due to the randomness in the self play process
    # if this is the case, the samples are aggregated and the mean is taken
    # todo: add unit test
    def _make_experiences_consistent(self):
        if len(self.experiences) == 0:
            return

        original_shape = self.experiences[0].neural_network_input.shape
        search_probability_list_by_neural_network_input = defaultdict(lambda: [])
        outcome_list_by_neural_network_input = defaultdict(lambda: [])
        for experience in self.experiences:
            hashable_nn_input = tuple(experience.neural_network_input.reshape(-1))
            search_probability_list_by_neural_network_input[hashable_nn_input].append(experience.search_probabilities)
            outcome_list_by_neural_network_input[hashable_nn_input].append(experience.outcome)

        self.experiences.clear()
        for neural_network_input in search_probability_list_by_neural_network_input:
            search_probabilities = np.array(
                search_probability_list_by_neural_network_input[neural_network_input]).mean(axis=0)
            outcome = mean(outcome_list_by_neural_network_input[neural_network_input])
            self.experiences.append(
                ReplayBufferExperience(
                    neural_network_input=np.array(neural_network_input).reshape(original_shape),
                    search_probabilities=search_probabilities,
                    outcome=outcome
                ))

    def get_experiences_from_indices(
            self,
            indices: list[int]
    ) -> list[ReplayBufferExperience]:
        return [self.experiences[i] for i in indices]
