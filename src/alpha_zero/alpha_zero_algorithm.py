import copy
import random
import time
from logging import Logger
from typing import Literal

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from agents.alpha_zero_agent import AlphaZeroAgent
from alpha_zero.replay_buffer_experience import ReplayBufferExperience
from constants.hyper_parameters import *
from game_logic.two_player_game import TwoPlayerGame


class AlphaZero:

    def __init__(self,
                 name_for_saving: str,
                 logger: Logger
                 ):
        self.NAME_FOR_SAVING = name_for_saving
        self.LOGGER = logger

    def start_training_pipeline(
            self,
            start_version: int,
            game: TwoPlayerGame,
            parallel: bool = False
    ) -> None:

        agent = AlphaZeroAgent(
            logger=self.LOGGER,
            player_number=1,
            version=start_version,
            seconds_per_move=SCNDS_PER_MOVE_TRAINING,
            game=game,
            name_for_saving=self.NAME_FOR_SAVING
        )

        while True:
            if parallel:
                self._self_play_parallel(agent, game)
            else:
                self._self_play_sequential(agent, game)
            agent.train()
            agent.version = agent.version + 1
            agent = self._evaluate_version(agent)

    # region Self Play

    def _self_play_sequential(
            self,
            agent: AlphaZeroAgent,
            game: TwoPlayerGame
    ) -> None:
        start_time = time.time()
        self._self_play_process(agent, game)
        print(f"\r\tSelf play completed\t time for Self play: {(time.time() - start_time)} seconds,"
              f"\t size replay buffer: {len(agent.REPLAY_BUFFER)}")

    def _self_play_parallel(
            self,
            agent: AlphaZeroAgent,
            game: TwoPlayerGame
    ) -> None:

        start_time = time.time()
        pool = Pool(processes=NR_PROCESSES_ON_CPU)

        args = [(agent.clone(), game.clone()) for _ in range(NR_PROCESSES_ON_CPU)]
        pool.map(self._self_play_process, args)
        pool.close()
        pool.join()

        print(f"\r\tSelf play completed\t time for Self play: {(time.time() - start_time)} seconds,"
              f"\t size replay buffer: {len(agent.REPLAY_BUFFER)}")

    def _self_play_process(
            self,
            agent_1: AlphaZeroAgent,
            game: TwoPlayerGame
    ) -> None:

        start_time = time.time()
        agent_2 = self._prepare_second_agent(agent_1)
        agents = [agent_1, agent_2]
        for agent in agents:
            agent.MCTS.set_training_mode_on()

        for episode in range(1, NUMBER_GAMES_PER_SELF_PLAY + 1):
            self._reset_agents(agent_1, agent_2)
            current_agent = agents[random.randint(0, 1)]

            experience_list = self._play_game(
                game=game,
                agent_1=agent_1,
                agent_2=agent_2,
                current_agent=current_agent
            )
            self._add_outcome_of_game_to_experience(experience_list, game.winner)
            self._store_results_in_replay_buffer(agent=agent_1, experience_list=experience_list)

            print(f"\r\tSelf play - {episode / NUMBER_GAMES_PER_SELF_PLAY * 100}% done"
                  f"\t time used so far: {(time.time() - start_time)} seconds"
                  f"\t size replay buffer: {len(agent_1.REPLAY_BUFFER)}",
                  end="")
        agent_1.REPLAY_BUFFER.consistency_check()
        for agent in agents:
            agent.MCTS.set_training_mode_off()

    def _prepare_second_agent(
            self,
            agent_1: AlphaZeroAgent
    ) -> AlphaZeroAgent:

        agent_2 = agent_1.clone()
        agent_2.set_player(-1)
        return agent_2

    def _reset_agents(
            self,
            agent_1: AlphaZeroAgent,
            agent_2: AlphaZeroAgent
    ) -> None:

        for agent in [agent_1, agent_2]:
            agent.MCTS.reset()

    def _play_game(
            self,
            game: TwoPlayerGame,
            agent_1: AlphaZeroAgent,
            agent_2: AlphaZeroAgent,
            current_agent: AlphaZeroAgent
    ) -> list[ReplayBufferExperience]:

        step = 0
        experience_list = []
        while game.winner is None:
            step += 1
            if len(game.feasible_actions) == 0:
                game.winner = 0
                break
            action = current_agent.compute_action()
            experience_list.append(self._create_experience(current_agent=current_agent, game=game))
            game.step_if_feasible(action, current_agent.player_number)
            current_agent = agent_1 if current_agent is agent_2 else agent_2
        return experience_list

    def _create_experience(
            self,
            current_agent: AlphaZeroAgent,
            game: TwoPlayerGame
    ) -> ReplayBufferExperience:
        """
        outcome of experience is not available yet and has to be set later
        """
        search_probabilities = self._compute_search_probabilities(current_agent, game)
        neural_network_input = current_agent.NETWORK_MANAGER.prepare_nn_input(
            copy.deepcopy(game.all_board_states),
            game.STATE_SHAPE,
            current_agent.player_number
        ).squeeze(0)  # todo: what does squeeze do?

        experience = ReplayBufferExperience(
            neural_network_input=neural_network_input,
            search_probabilities=search_probabilities,
            outcome=None
        )
        return experience

    def _compute_search_probabilities(
            self,
            agent: AlphaZeroAgent,
            game: TwoPlayerGame
    ) -> np.array:
        search_probabilities = np.zeros(game.NO_ACTIONS)
        for child in agent.MCTS.current_root.children:
            action = child.ACTION_BEFORE_STATE
            prob = child.visit_count / child.FATHER.visit_count
            search_probabilities[action] = prob
        return search_probabilities

    def _add_outcome_of_game_to_experience(
            self,
            experience_list: list[ReplayBufferExperience],
            outcome: Literal[-1, 0, 1]
    ) -> None:
        for experience in experience_list:
            experience.outcome = outcome

    def _store_results_in_replay_buffer(
            self,
            agent: AlphaZeroAgent,
            experience_list
    ) -> None:
        for experience in experience_list:
            agent.REPLAY_BUFFER.add_experience(experience)

    # endregion Self Play

    # evaluate version by playing vs previous version
    def _evaluate_version(self, agent: AlphaZeroAgent):
        """
        evaluate version by playing vs previous version
        """
        start_time = time.time()
        # play vs older version for evaluation
        agent_old = AlphaZeroAgent(
            logger=self.LOGGER,
            player_number=1,
            version=agent.version - 1,
            seconds_per_move=SCNDS_PER_MOVE_TRAINING,
            game=self.GAME,
            name_for_saving=self.NAME_FOR_SAVING
        )

        number_wins_new_agent = 0
        number_ties = 0
        for i in range(1, NUMBER_GAMES_VS_OLD_VERSION + 1):
            outcome = play_game_return_winner(agent, agent_old, self.GAME)
            if outcome == 1:
                number_wins_new_agent += 1
            elif outcome == 0:
                number_ties += 1
            win_prob = number_wins_new_agent / i * 100
            tie_prob = number_ties / i * 100
            print(
                "\r\t{} games completed\t time so far: {} seconds\twin probability: {}%\t tie probability: {}%".format(
                    i,
                    (
                            time.time() - start_time),
                    win_prob,
                    tie_prob),
                end="")
        print()
        # ties count as wins times a weight, otherwise it is too hard to be accepted because of the high tie probability in some games
        if number_wins_new_agent + number_ties * WEIGHT_FOR_TIES_IN_EVALUATION >= WIN_PERCENTAGE * NUMBER_GAMES_VS_OLD_VERSION / 100:
            # accept new version
            # reset replay buffer
            # agent.replay_buffer.reset()
            agent.save(agent.version)

            print("version {} accepted with win probability: {}% and tie probability: {}%".format(agent.version,
                                                                                                  number_wins_new_agent / NUMBER_GAMES_VS_OLD_VERSION * 100,
                                                                                                  number_ties / NUMBER_GAMES_VS_OLD_VERSION * 100))
        else:
            print("version {} refused with win probability: {}% and tie probability: {}%".format(agent.version,
                                                                                                 number_wins_new_agent / NUMBER_GAMES_VS_OLD_VERSION * 100,
                                                                                                 number_ties / NUMBER_GAMES_VS_OLD_VERSION * 100))
            # go back to network of previous version but keep replay buffer
            agent.version = agent.version - 1
            agent.network = NeuralNetwork(version=agent.version, nr_actions=self.GAME.NR_ACTIONS,
                                          state_shape=self.GAME.state_shape,
                                          name_for_saving=agent.NAME_FOR_SAVING)
        # always save replaybuffer
        agent.REPLAY_BUFFER.save_to_file(agent.version)
        print("\tEvaluation completed\t time for evaluation: {} seconds".format((time.time() - start_time)))
        print(
            "__________________________________________________________________________________________________________________________")
        return agent

    # # play a game and return true if agent wins
    def play_game_return_winner(agent, agent_old, game):
        agents = [agent, agent_old]
        crnt_agent = agents[game.random.randint(0, 1)]
        while game.winner is None:
            if len(game.feasible_actions) == 0:
                game.winner = 0
                break
            action = crnt_agent.compute_action(game, False)
            game.step_if_feasible(action, crnt_agent.player)
            crnt_agent = agent if crnt_agent is agent_old else agent_old
        return game.winner * agent.player_number

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
