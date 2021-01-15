import random
import time


from mcts import MCTS
from neural_network import Neural_network
from replay_buffer import Replay_buffer

from hyperparameters import *


class Agent():
    def __init__(self, type, player, seed, version=0, scnds_per_move=1, game=None, replay_buffer=None,
                 name_for_saving=None):
        self.type = type
        self.player = player
        self.game = game
        self.name = type
        self.random = random
        self.random.seed(seed)
        if type == "alphaZero":
            self.name += "_{}_".format(version)
        if type == "mcts":
            self.mcts = MCTS(scnds_per_move, game, player, seed=seed)
        if type == "alphaZero":
            self.seed = seed
            self.network = Neural_network(version, game.nr_actions, game.state_shape, name_for_saving)
            self.replay_buffer = replay_buffer
            if self.replay_buffer is None:
                self.replay_buffer = Replay_buffer(version, name_for_saving, seed)
            self.version = version
            self.mcts = MCTS(scnds_per_move=scnds_per_move, game=game, player=self.player,seed=seed, network=self.network)
            self.scnds_per_move = scnds_per_move
            self.name_for_saving = name_for_saving

    def reset_mcts(self):
        self.mcts.reset()

    def clone(self):
        clone = Agent(type=self.type, player=self.player, seed=self.seed, version=self.version,
                      scnds_per_move=self.scnds_per_move, game=self.game,
                      replay_buffer=self.replay_buffer, name_for_saving=self.name_for_saving)
        # clone shall have its own mcts
        clone.mcts = MCTS(self.scnds_per_move, self.game, self.player, self.seed, self.network)
        return clone

    def compute_action(self, game, training=False):
        game.user_action = None
        if self.type == "random":
            return game.random.choice(game.feasible_actions)
        elif self.type == "user":
            while game.user_action is None:
                game.gui.refresh_picture(game.board)
                time.sleep(0.01)
            action = game.user_action
            game.user_action = None
            return action
        elif self.type == "mcts" or self.type == "alphaZero":
            return self.mcts.step(training)

    def train(self):
        start_time = time.time()
        total_value_loss = 0
        total_policy_loss = 0
        # take the validation indices out before, so that training does not consider them
        size_rpb = len(self.replay_buffer.outcomes)
        indices = [i for i in range(size_rpb)]
        # hold at most 1 percent of the data back for validation
        validation_indices = self.random.sample(indices, k=int(min(NUMBER_OF_BATCHES_VALIDATION*BATCH_SIZE, size_rpb/10)))
        training_indices = [i for i in indices if i not in validation_indices]
        for episode in range(1, NUMBER_OF_BATCHES_TRAINING+1):
            # sample a mini batch
            nn_inputs, search_probabilities, outcomes = self.replay_buffer.sample(BATCH_SIZE, indices=training_indices)
            # compute tensors and send to device
            input_tensors = torch.from_numpy(nn_inputs).float().to(DEVICE)
            search_probabilities = torch.from_numpy(search_probabilities).float().to(DEVICE)
            outcomes = torch.from_numpy(outcomes).float().to(DEVICE)
            # compute output of network
            values, move_probabilities = self.network.forward(input_tensors, self.player)
            values = values.squeeze()
            # compute loss
            value_loss = ((values - outcomes) ** 2).mean()
            policy_loss = -(search_probabilities * torch.log(move_probabilities)).mean()
            loss = value_loss + policy_loss
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            # train
            optimizer = torch.optim.SGD(self.network.parameters(), lr=self.compute_learning_rate(episode),
                                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if episode == 1:
                print(
                    'First episode \t average value loss: {} \t average policy loss: {}'.format(
                        total_value_loss / episode
                        , total_policy_loss / episode))
            print(
                '\r\t# batches trained {} ( out of {}) \t time for training: {} seconds \t average value loss: {} \t average policy loss: {}'.format(
                    episode,
                    NUMBER_OF_BATCHES_TRAINING,
                    (time.time() - start_time),
                    total_value_loss / episode
                , total_policy_loss / episode), end="")
        print(
            '\r\tTraining completed \t time for training: {} seconds \t average value loss: {} \t average policy loss: {}'.format(
                (time.time() - start_time), total_value_loss / NUMBER_OF_BATCHES_TRAINING
            , total_policy_loss / NUMBER_OF_BATCHES_TRAINING))
        self.validate(validation_indices)

    def validate(self, validation_indices):
        start_time = time.time()
        # here we do not sample, instead we take everything from the held out data
        nn_inputs_numpy, search_probabilities_numpy, outcomes_numpy = self.replay_buffer.sample(len(validation_indices), indices=validation_indices)
        value_loss = 0
        policy_loss = 0
        # we cannot feed all the data at once into the neural network because of the gpu memory
        slice_len = len(validation_indices)//NUMBER_OF_BATCHES_VALIDATION
        for i in range(NUMBER_OF_BATCHES_VALIDATION):
            # compute tensors of correct slices and send to device
            input_tensors = torch.from_numpy(nn_inputs_numpy[i*slice_len:(i+1)*slice_len]).float().to(DEVICE)
            search_probabilities = torch.from_numpy(search_probabilities_numpy[i*slice_len:(i+1)*slice_len]).float().to(DEVICE)
            outcomes = torch.from_numpy(outcomes_numpy[i*slice_len:(i+1)*slice_len]).float().to(DEVICE)
            # compute output of network
            values, move_probabilities = self.network.forward(input_tensors, self.player)
            values = values.squeeze()
            # compute loss
            value_loss += ((values - outcomes) ** 2).mean().item()
            policy_loss += -(search_probabilities * torch.log(move_probabilities)).mean().item()
        print(
            '\r\tValidation completed \t time for validation: {} seconds \t average value loss: {} \t average policy loss: {}'.format(
                (time.time() - start_time), value_loss/NUMBER_OF_BATCHES_VALIDATION , policy_loss/NUMBER_OF_BATCHES_VALIDATION ))

    def save(self, version):
        self.network.save_model(self.version)
        self.replay_buffer.save_to_file(version)

    def compute_learning_rate(self, episode):
        if episode * BATCH_SIZE < 400000:
            return 0.01
        if episode * BATCH_SIZE < 600000:
            return 0.001
        return 0.0001
