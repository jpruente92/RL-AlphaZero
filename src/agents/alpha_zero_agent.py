from logging import Logger
from typing import Literal

from agents.base_agent import BaseAgent
from game_logic.two_player_game import TwoPlayerGame
from neural_network.neural_network_torch.network_manager_torch import NetworkManagerTorch
from alpha_zero.replay_buffer import ReplayBuffer
from monte_carlo_tree_search.mcts_with_nn import MCTSWithNeuralNetwork


class AlphaZeroAgent(BaseAgent):

    def __init__(
            self,
            logger: Logger,
            game: TwoPlayerGame,
            player_number: Literal[-1, 1],
            version=0,
            seconds_per_move=1,
            replay_buffer=None,
            name_for_saving="Connect_4"  # todo: make it dependent from the game
    ):
        super().__init__(
            logger=logger,
            name=f"AlphaZero_{version}",
            player_number=player_number,
            game=game
        )

        self.SECONDS_PER_MOVE = seconds_per_move

        self.NETWORK_MANAGER = NetworkManagerTorch(
            logger=logger,
            name_for_saving=name_for_saving,
            version=version,
            no_actions=game.NO_ACTIONS,
            state_shape=game.STATE_SHAPE
        )
        self.MCTS = MCTSWithNeuralNetwork(
            logger=logger,
            seconds_per_move=self.SECONDS_PER_MOVE,
            game=self.GAME,
            player_number=self.player_number,
            network_manager=self.NETWORK_MANAGER
        )

        self.VERSION = version
        self.NAME_FOR_SAVING = name_for_saving

        self.REPLAY_BUFFER = replay_buffer
        if self.REPLAY_BUFFER is None:
            self.REPLAY_BUFFER = ReplayBuffer(name_for_saving)

    # region Public Methods

    def set_player(self, player_number: Literal[-1, 1]):
        self.player_number = player_number
        self.MCTS.PLAYER_NUMBER = player_number

    def compute_action(
            self
    ) -> int:
        self.GAME.user_action = None
        return self.MCTS.step()

    def clone(self):
        clone = AlphaZeroAgent(
            player_number=self.player_number,
            version=self.VERSION,
            seconds_per_move=self.SECONDS_PER_MOVE,
            game=self.GAME,
            replay_buffer=self.REPLAY_BUFFER,
            name_for_saving=self.NAME_FOR_SAVING
        )
        return clone

    # endregion Public Methods

    # todo: refactor from here
    # region Training
    def train(self):
        start_time = time.time()
        total_value_loss = 0
        total_policy_loss = 0
        # take the validation indices out before, so that training does not consider them
        size_rpb = len(self.replay_buffer.outcomes)
        indices = [i for i in range(size_rpb)]
        # hold at most 1 percent of the data back for validation
        validation_indices = self.random.sample(indices,
                                                k=int(min(NUMBER_OF_BATCHES_VALIDATION * BATCH_SIZE, size_rpb / 10)))
        training_indices = [i for i in indices if i not in validation_indices]
        self.validate(validation_indices, "before training")
        for episode in range(1, NUMBER_OF_BATCHES_TRAINING + 1):
            # sample a mini batch
            nn_inputs, search_probabilities, outcomes = self.replay_buffer.sample(BATCH_SIZE, indices=training_indices)
            # compute tensors and send to device
            input_tensors = torch.from_numpy(nn_inputs).float().to(DEVICE)
            search_probabilities = torch.from_numpy(search_probabilities).float().to(DEVICE)
            outcomes = torch.from_numpy(outcomes).float().to(DEVICE)
            # compute output of network
            values, move_probabilities = self.network.forward(input_tensors)
            values = values.squeeze()
            # compute loss
            value_loss = ((values - outcomes) ** 2).mean()
            policy_loss = self.policy_loss(search_probabilities, move_probabilities)
            loss = value_loss + WEIGHT_POLICY_LOSS * policy_loss
            total_value_loss += value_loss.item()
            total_policy_loss += policy_loss.item()
            # train
            optimizer = torch.optim.SGD(self.network.parameters(), lr=self.compute_learning_rate(episode),
                                        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
        self.validate(validation_indices, "after training")

    def validate(self, validation_indices, string):
        start_time = time.time()
        # here we do not sample, instead we take everything from the held out data
        nn_inputs_numpy, search_probabilities_numpy, outcomes_numpy = self.replay_buffer.sample(len(validation_indices),
                                                                                                indices=validation_indices)
        value_loss = 0
        policy_loss = 0
        # we cannot feed all the data at once into the neural network because of the gpu memory
        slice_len = len(validation_indices) // NUMBER_OF_BATCHES_VALIDATION
        for i in range(NUMBER_OF_BATCHES_VALIDATION):
            # compute tensors of correct slices and send to device
            input_tensors = torch.from_numpy(nn_inputs_numpy[i * slice_len:(i + 1) * slice_len]).float().to(DEVICE)
            search_probabilities = torch.from_numpy(
                search_probabilities_numpy[i * slice_len:(i + 1) * slice_len]).float().to(DEVICE)
            outcomes = torch.from_numpy(outcomes_numpy[i * slice_len:(i + 1) * slice_len]).float().to(DEVICE)
            # compute output of network
            values, move_probabilities = self.network.forward(input_tensors)
            values = values.squeeze()
            # compute loss
            value_loss += ((values - outcomes) ** 2).mean().item()
            policy_loss += self.policy_loss(search_probabilities, move_probabilities).item()
        print(
            '\r\tValidation {} completed \t time for validation: {} seconds \t average value loss: {} \t average policy loss: {}'.format(
                string, (time.time() - start_time), value_loss / NUMBER_OF_BATCHES_VALIDATION,
                                                    policy_loss / NUMBER_OF_BATCHES_VALIDATION))

    def save(self, version):
        self.network.save_model(self.version)
        self.replay_buffer.save_to_file(version)

    def compute_learning_rate(self, episode):
        if episode * BATCH_SIZE < 400000:
            return 0.01
        if episode * BATCH_SIZE < 600000:
            return 0.001
        return 0.0001

    def policy_loss(self, search_probabilities, move_probabilities):
        return -(search_probabilities * torch.log(move_probabilities)).mean(axis=0).sum()
        # return ((search_probabilities - move_probabilities)**2).mean(axis=0).sum()

    # endregion Training
