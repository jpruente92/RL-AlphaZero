from neural_network import Neural_network
from replay_buffer import Replay_buffer


class Agent():
    def __init__(self,version):
        if version >-1:
            self.network = Neural_network(version)
            self.replay_buffer=Replay_buffer(version)


    def train(self):
        pass

    def save(self,version):
        pass
