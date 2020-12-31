from collections import deque, namedtuple
from random import random

from hyperparameters import *


class Replay_buffer():
    def __init__(self, version):
        self.memory = deque(maxlen=BUFFER_SIZE)
        self.game = namedtuple("Experience",
                               field_names=["board_configurations", "actions", "outcome", "starting_player"])
        self.seed = random.seed(SEED)
        if version > 0:
            self.load_from_file("replay_buffers_version_" + str(version))

    def save_to_file(self, version):
        pass

    def load_from_file(self, filename):
        pass

    def sample(self):
        pass
