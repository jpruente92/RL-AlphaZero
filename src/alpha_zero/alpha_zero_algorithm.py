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
from game_logic.game_state import GameState
from game_logic.two_player_game import TwoPlayerGame
from monte_carlo_tree_search.node import Node


class AlphaZero:

    def __init__(self,
                 logger: Logger
                 ):
        self.LOGGER = logger

    def start_training_pipeline(
            self,
            alpha_zero_agent: AlphaZeroAgent,
            game: TwoPlayerGame,
            parallel: bool
    ) -> None:

        current_agent = alpha_zero_agent
        while True:
            self._self_play(
                alpha_zero_agent=current_agent,
                game=game,
                parallel=parallel
            )
            trained_agent = self._training(alpha_zero_agent=current_agent)
            current_agent = self._evaluate_agent(
                alpha_zero_agent=trained_agent,
                alpha_zero_agent_previous=current_agent,
                game=game
            )

    # region Self Play

    def _self_play(
            self,
            alpha_zero_agent: AlphaZeroAgent,
            game: TwoPlayerGame,
            parallel: bool
    ) -> None:
        start_time = time.time()
        if parallel:
            self._self_play_parallel(alpha_zero_agent, game)
        else:
            self._self_play_sequential(alpha_zero_agent, game)

        alpha_zero_agent.REPLAY_BUFFER.save_to_file(alpha_zero_agent.VERSION)

        self.LOGGER.info(f"\t\tSelf play completed"
                         f"\t\ttime for Self play: {(time.time() - start_time):.2f} seconds"
                         f"\t\tsize replay buffer: {len(alpha_zero_agent.REPLAY_BUFFER)}")

    def _self_play_sequential(
            self,
            agent: AlphaZeroAgent,
            game: TwoPlayerGame
    ) -> None:
        self._self_play_process(agent, game)

    def _self_play_parallel(
            self,
            alpha_zero_agent: AlphaZeroAgent,
            game: TwoPlayerGame
    ) -> None:

        pool = Pool(processes=NR_PROCESSES_ON_CPU)

        agents = [alpha_zero_agent.clone() for _ in range(NR_PROCESSES_ON_CPU)]
        args = [(agent, game) for agent in agents]
        pool.map(self._self_play_process, args)
        pool.close()
        pool.join()

        for agent in agents:
            alpha_zero_agent.REPLAY_BUFFER.add_experiences(agent.REPLAY_BUFFER.experiences)

    # todo: add unit test
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
            index_starting_agent = random.randint(0, 1)

            experience_list = self._play_game(
                game=game,
                agent_1=agent_1,
                agent_2=agent_2,
                index_starting_agent=index_starting_agent
            )
            self._store_results_in_replay_buffer(agent=agent_1, experience_list=experience_list)

            print(f"\r\tSelf play - "
                  f"\t\t{episode / NUMBER_GAMES_PER_SELF_PLAY * 100:.2f}% done"
                  f"\t\ttime used so far: {(time.time() - start_time):.2f} seconds"
                  f"\t\tsize replay buffer: {len(agent_1.REPLAY_BUFFER)}",
                  end="")
        print("\r", end="")

        for agent in agents:
            agent.MCTS.set_training_mode_off()

    def _prepare_second_agent(
            self,
            agent_1: AlphaZeroAgent
    ) -> AlphaZeroAgent:

        agent_2 = agent_1.copy()
        agent_2.set_player(-agent_1.player_number)
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
            index_starting_agent: int
    ) -> list[ReplayBufferExperience]:
        current_agent = [agent_1, agent_2][index_starting_agent]
        game_state = game.start_game(
            agent_1=agent_1,
            agent_2=agent_2,
            index_starting_agent=index_starting_agent
        )
        all_roots = self._compute_all_roots_in_correct_order(agent_1)
        experience_list = self._create_experience_list(
            current_agent=current_agent,
            all_roots=all_roots,
            game=game,
            game_state=game_state
        )
        return experience_list

    def _compute_all_roots_in_correct_order(self, agent_1):
        all_roots = agent_1.MCTS.all_roots
        all_roots = sorted(all_roots, key=lambda root: root.DEPTH)
        for index, node in enumerate(all_roots):
            assert index == node.DEPTH
        return all_roots

    def _create_experience_list(
            self,
            current_agent: AlphaZeroAgent,
            all_roots: list[Node],
            game: TwoPlayerGame,
            game_state: GameState
    ) -> list[ReplayBufferExperience]:
        experience_list = []

        winner = game_state.winner
        assert winner is not None

        all_board_states = []
        for node in all_roots:
            all_board_states.append(node.GAME_STATE.board)
            search_probabilities = self._compute_search_probabilities(game=game, current_node=node)
            neural_network_input = current_agent.NETWORK_MANAGER.prepare_nn_input(
                copy.deepcopy(all_board_states),
                node.GAME_STATE.player_number_to_move
            )

            experience = ReplayBufferExperience(
                neural_network_input=neural_network_input,
                search_probabilities=search_probabilities,
                outcome=winner
            )
            experience_list.append(experience)
        return experience_list

    def _compute_search_probabilities(
            self,
            game: TwoPlayerGame,
            current_node: Node
    ) -> np.array:
        search_probabilities = np.zeros(game.NO_ACTIONS)
        for child in current_node.children:
            action = child.ACTION_BEFORE_STATE
            prob = child.visit_count / child.FATHER.visit_count
            search_probabilities[action] = prob
        return search_probabilities

    def _store_results_in_replay_buffer(
            self,
            agent: AlphaZeroAgent,
            experience_list: list[ReplayBufferExperience]
    ) -> None:
        for experience in experience_list:
            agent.REPLAY_BUFFER.add_experience(experience)

    # endregion Self Play

    # region Training

    def _training(
            self,
            alpha_zero_agent: AlphaZeroAgent
    ) -> AlphaZeroAgent:
        start_time = time.time()
        total_value_loss = 0
        total_policy_loss = 0

        trained_agent = alpha_zero_agent.clone()

        training_indices, validation_indices = self._compute_indices(trained_agent)

        self._log_validation_loss(
            validation_indices=validation_indices,
            time_point="before training",
            alpha_zero_agent=trained_agent
        )

        for episode in range(1, NUMBER_OF_BATCHES_TRAINING + 1):
            sample_indices = random.sample(training_indices, k=min(BATCH_SIZE, len(training_indices)))
            experiences_of_batch = \
                trained_agent.REPLAY_BUFFER.get_experiences_from_indices(sample_indices)

            policy_loss, value_loss = trained_agent.NETWORK_MANAGER.train_batch(
                learning_rate=self._compute_learning_rate(episode),
                experiences_of_batch=experiences_of_batch,
            )
            total_policy_loss += policy_loss
            total_value_loss += value_loss

            print(f'\r\t\t# batches trained {episode} ( out of {NUMBER_OF_BATCHES_TRAINING})'
                  f'\t\ttime for training: {(time.time() - start_time):.2f} seconds'
                  f'\t\taverage value loss: {total_value_loss / episode:.2f}'
                  f'\t\taverage policy loss: {total_policy_loss / episode:.2f}', end="")
        print("\r", end="")
        self.LOGGER.info(f"\t\tTraining completed"
                         f"\t\ttime for training: {(time.time() - start_time):.2f} seconds"
                         f"\t\taverage value loss: {total_value_loss / NUMBER_OF_BATCHES_TRAINING:.2f}"
                         f"\t\taverage policy loss: {total_policy_loss / NUMBER_OF_BATCHES_TRAINING:.2f}"
                         )

        self._log_validation_loss(
            validation_indices=validation_indices,
            time_point="after training",
            alpha_zero_agent=trained_agent
        )
        return trained_agent

    def _compute_indices(
            self,
            alpha_zero_agent: AlphaZeroAgent
    ) -> (list[int], list[int]):
        size_replay_buffer = len(alpha_zero_agent.REPLAY_BUFFER)
        indices = [i for i in range(size_replay_buffer)]
        no_validation_indices = size_replay_buffer / PERCENTAGE_DATA_FOR_VALIDATION
        validation_indices = random.sample(
            indices,
            k=int(no_validation_indices)
        )
        training_indices = [i for i in indices if i not in validation_indices]
        return training_indices, validation_indices

    def _log_validation_loss(
            self,
            validation_indices: list[int],
            time_point: str,
            alpha_zero_agent: AlphaZeroAgent
    ) -> None:
        start_time = time.time()

        number_of_batches_validation = len(validation_indices) // BATCH_SIZE

        experiences = alpha_zero_agent.REPLAY_BUFFER.get_experiences_from_indices(indices=validation_indices)
        total_policy_loss, total_value_loss = alpha_zero_agent.NETWORK_MANAGER.compute_loss_of_batch(
            alpha_zero_agent=alpha_zero_agent,
            experiences=experiences,
            number_of_batches_validation=number_of_batches_validation
        )
        self.LOGGER.info(
            f'Validation {time_point} completed '
            f'\t\ttime for validation: {(time.time() - start_time):.2f} seconds'
            f'\t\taverage value loss per batch: {total_value_loss / number_of_batches_validation:.2f}'
            f'\t\taverage policy loss per batch: {total_policy_loss / number_of_batches_validation:.2f}'
        )

    def _compute_learning_rate(
            self,
            episode: int
    ) -> float:
        if episode * BATCH_SIZE < 400_000:
            return 0.01
        if episode * BATCH_SIZE < 600_000:
            return 0.001
        return 0.0001

    # endregion Training

    # region Evaluation

    def _evaluate_agent(
            self,
            alpha_zero_agent: AlphaZeroAgent,
            alpha_zero_agent_previous: AlphaZeroAgent,
            game: TwoPlayerGame
    ) -> AlphaZeroAgent:
        start_time = time.time()

        alpha_zero_agent_previous.player_number = - alpha_zero_agent.player_number

        number_ties, number_wins_new_agent = \
            self._compute_win_and_tie_probability(
                alpha_zero_agent=alpha_zero_agent,
                alpha_zero_agent_previous=alpha_zero_agent_previous,
                game=game
            )

        if self._accept_version(
                number_wins_new_agent=number_wins_new_agent,
                number_ties=number_ties
        ):
            alpha_zero_agent.VERSION = alpha_zero_agent.VERSION + 1
            alpha_zero_agent.REPLAY_BUFFER.reset()
            alpha_zero_agent.NETWORK_MANAGER.save_model(alpha_zero_agent.VERSION)

            self.LOGGER.info(
                f"\t\tVersion {alpha_zero_agent.VERSION} accepted"
                f" with win probability: {100 * number_wins_new_agent / NUMBER_GAMES_VS_OLD_VERSION:.2f}%"
                f" and tie probability: {100 * number_ties / NUMBER_GAMES_VS_OLD_VERSION:.2f}%"
            )
        else:
            self.LOGGER.info(
                f"\t\tVersion {alpha_zero_agent.VERSION + 1} refused"
                f" with win probability: {100 * number_wins_new_agent / NUMBER_GAMES_VS_OLD_VERSION:.2f}%"
                f" and tie probability: {100 * number_ties / NUMBER_GAMES_VS_OLD_VERSION:.2f}%"
            )
            alpha_zero_agent.NETWORK_MANAGER.reset_neural_network()

        self.LOGGER.info(f"\t\tEvaluation completed"
                         f"\t\ttime for evaluation: {time.time() - start_time:.2f} seconds")
        return alpha_zero_agent

    def _accept_version(
            self,
            number_wins_new_agent: int,
            number_ties: int,
    ) -> bool:
        bonus_for_ties = number_ties * WEIGHT_FOR_TIES_IN_EVALUATION
        return number_wins_new_agent + bonus_for_ties >= WIN_PERCENTAGE * NUMBER_GAMES_VS_OLD_VERSION / 100

    def _compute_win_and_tie_probability(
            self,
            alpha_zero_agent: AlphaZeroAgent,
            alpha_zero_agent_previous: AlphaZeroAgent,
            game: TwoPlayerGame
    ) -> (int, int):

        start_time = time.time()
        number_wins_new_agent = 0
        number_ties = 0
        for i in range(1, NUMBER_GAMES_VS_OLD_VERSION + 1):
            winner = self._play_game_return_winning_player_number(
                alpha_zero_agent=alpha_zero_agent,
                alpha_zero_agent_previous=alpha_zero_agent_previous,
                game=game
            )
            if winner == 1:
                number_wins_new_agent += 1
            elif winner == 0:
                number_ties += 1
            win_probability = number_wins_new_agent / i * 100
            tie_probability = number_ties / i * 100
            print(f"\r\t\t{100 * i / NUMBER_GAMES_VS_OLD_VERSION:.2f} % of games completed"
                  f"\t\ttime so far: {time.time() - start_time:.2f} seconds"
                  f"\t\twin probability: {win_probability:.2f}%"
                  f"\t\t tie probability: {tie_probability:.2f}%",
                  end="")
        print("\r", end="")
        return number_ties, number_wins_new_agent

    def _play_game_return_winning_player_number(
            self,
            alpha_zero_agent: AlphaZeroAgent,
            alpha_zero_agent_previous: AlphaZeroAgent,
            game: TwoPlayerGame
    ) -> Literal[-1, 0, 1]:

        index_starting_agent = random.randint(0, 1)
        game_state = game.start_game(
            agent_1=alpha_zero_agent,
            agent_2=alpha_zero_agent_previous,
            index_starting_agent=index_starting_agent
        )
        return game_state.winner

    # endregion Evaluation
