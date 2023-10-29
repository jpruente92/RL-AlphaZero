import dataclasses
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class ReplayBufferExperience:
    neural_network_input: np.array
    search_probabilities: np.array
    outcome: Optional[float]

    def to_dict(self):
        return dataclasses.asdict(self)

    def __eq__(self, other):
        return np.array_equal(self.neural_network_input, other.neural_network_input) \
               and np.array_equal(self.search_probabilities, other.search_probabilities) \
               and self.outcome == other.outcome

