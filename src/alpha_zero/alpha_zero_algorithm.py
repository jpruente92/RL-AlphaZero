import copy
import random
import time
from logging import Logger
from multiprocessing import Pool

import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool

from agents.alpha_zero_agent import AlphaZeroAgent
from alpha_zero.replay_buffer_experience import ReplayBufferExperience
from constants.hyper_parameters import *
from game_logic.two_player_game import TwoPlayerGame


# def self_play_parallel(agent, game):
#     start_time = time.time()
#     pool = Pool(processes=NR_PROCESSES_ON_CPU)
#
#     args = [(agent.clone(), game.clone(), True) for i in range(NR_PROCESSES_ON_CPU)]
#     pool.map(self_play_process, args)
#     pool.close()
#     pool.join()
#
#     print("\r\tSelf play completed\t time for Self play: {} seconds".format((time.time() - start_time)))
#
#
# # play a game and return true if agent wins
# def play_game_return_winner(agent, agent_old, game):
#     game.reset()
#     agents = [agent, agent_old]
#     crnt_agent = agents[game.random.randint(0, 1)]
#     while game.winner is None:
#         if len(game.feasible_actions) == 0:
#             game.winner = 0
#             break
#         action = crnt_agent.compute_action(game, False)
#         game.step_if_feasible(action, crnt_agent.player)
#         crnt_agent = agent if crnt_agent is agent_old else agent_old
#     return game.winner * agent.PLAYER_NUMBER
#
#
# # evaluate version by playing vs previous version
# def evaluate(agent, game):
#     start_time = time.time()
#     # play vs older version for evaluation
#     # agent_old = Agent(type="alphaZero", player=-1, seed=agent.seed, version=agent.version - 1,
#     #                   scnds_per_move=SCNDS_PER_MOVE_TRAINING,
#     #                   game=game, name_for_saving=agent.name_for_saving)
#     number_wins_new_agent = 0
#     number_ties = 0
#     for i in range(1, NUMBER_GAMES_VS_OLD_VERSION + 1):
#         outcome = play_game_return_winner(agent, agent_old, game)
#         if outcome == 1:
#             number_wins_new_agent += 1
#         elif outcome == 0:
#             number_ties += 1
#         win_prob = number_wins_new_agent / i * 100
#         tie_prob = number_ties / i * 100
#         print("\r\t{} games completed\t time so far: {} seconds\twin probability: {}%\t tie probability: {}%".format(i,
#                                                                                                                      (
#                                                                                                                              time.time() - start_time),
#                                                                                                                      win_prob,
#                                                                                                                      tie_prob),
#               end="")
#     print()
#     # ties count as wins times a weight, otherwise it is too hard to be accepted because of the high tie probability in some games
#     if number_wins_new_agent + number_ties * WEIGHT_FOR_TIES_IN_EVALUATION >= WIN_PERCENTAGE * NUMBER_GAMES_VS_OLD_VERSION / 100:
#         # accept new version
#         # reset replay buffer
#         # agent.replay_buffer.reset()
#         agent.save(agent.version)
#
#         print("version {} accepted with win probability: {}% and tie probability: {}%".format(agent.version,
#                                                                                               number_wins_new_agent / NUMBER_GAMES_VS_OLD_VERSION * 100,
#                                                                                               number_ties / NUMBER_GAMES_VS_OLD_VERSION * 100))
#     else:
#         print("version {} refused with win probability: {}% and tie probability: {}%".format(agent.version,
#                                                                                              number_wins_new_agent / NUMBER_GAMES_VS_OLD_VERSION * 100,
#                                                                                              number_ties / NUMBER_GAMES_VS_OLD_VERSION * 100))
#         # go back to network of previous version but keep replay buffer
#         agent.version = agent.version - 1
#         agent.network = NeuralNetwork(version=agent.version, nr_actions=game.NR_ACTIONS, state_shape=game.state_shape,
#                                       name_for_saving=agent.NAME_FOR_SAVING)
#     # always save replaybuffer
#     agent.replay_buffer.save_to_file(agent.version)
#     print("\tEvaluation completed\t time for evaluation: {} seconds".format((time.time() - start_time)))
#     print(
#         "__________________________________________________________________________________________________________________________")
#     return agent


class AlphaZero:

    def __init__(self,
                 game: TwoPlayerGame,
                 name_for_saving: str,
                 logger: Logger
                 ):
        self.GAME = game
        self.NAME_FOR_SAVING = name_for_saving
        self.LOGGER = logger

    def start_training_pipeline(self, start_version):
        agent = AlphaZeroAgent(
            player_number=1,
            version=start_version,
            seconds_per_move=SCNDS_PER_MOVE_TRAINING,
            game=self.GAME,
            name_for_saving=self.NAME_FOR_SAVING
        )

        while True:
            # play games to fill replay buffer
            self._self_play(agent)
            # learning from games to improve neural network
            agent.train()
            agent.version = agent.version + 1
            # evaluate version
            agent = evaluate(agent, game)

    def _self_play(self, agent):
        start_time = time.time()
        self._self_play_process(agent)
        print("\r\tSelf play completed\t time for Self play: {} seconds size replay buffer: {}".format(
            (time.time() - start_time), len(agent.replay_buffer)))

    def _self_play_process(
            self,
            agent_1: AlphaZeroAgent
    ) -> None:
        start_time = time.time()
        agent_2 = self._prepare_second_agent(agent_1)
        agents = [agent_1, agent_2]
        for agent in agents:
            agent.MCTS.set_training_mode_on()

        for episode in range(1, NUMBER_GAMES_PER_SELF_PLAY + 1):
            self._reset_game_and_agents(agent_1, agent_2)
            current_agent = agents[random.randint(0, 1)]

            step = 0
            experience_list = []
            while self.GAME.winner is None:
                step += 1
                if len(self.GAME.FEASIBLE_ACTIONS) == 0:
                    self.GAME.winner = 0
                    break
                action = current_agent.compute_action()
                experience_list.append(self._create_experience(current_agent=current_agent))
                self.GAME.step_if_feasible(action, current_agent.PLAYER_NUMBER)
                current_agent = agent_1 if current_agent is agent_2 else agent_2

            self._store_results_in_replay_buffer(agent=agent_1, experience_list=experience_list)

            print(f"\r\tSelf play - {episode / NUMBER_GAMES_PER_SELF_PLAY * 100}% done"
                  f"\t time used so far: {(time.time() - start_time)} seconds"
                  f"\t size replay buffer: {len(agent_1.REPLAY_BUFFER)}",
                  end="")
        agent_1.REPLAY_BUFFER.consistency_check()
        for agent in agents:
            agent.MCTS.set_training_mode_off()

    def _store_results_in_replay_buffer(
            self,
            agent: AlphaZeroAgent,
            experience_list
    ) -> None:
        for experience in experience_list:
            experience.outcome = self.GAME.winner
            agent.REPLAY_BUFFER.add_experience(experience)

    def _create_experience(self, current_agent):
        """
        outcome of experience is not available yet and has to be set later
        """
        search_probabilities = self._compute_search_probabilities(current_agent)
        neural_network_input = self._prepare_nn_input(
            copy.deepcopy(self.GAME.all_board_states), current_agent.PLAYER_NUMBER).squeeze(0)
        experience = ReplayBufferExperience(
            neural_network_input=neural_network_input,
            search_probabilities=search_probabilities,
            outcome=None
        )
        return experience

    def _reset_game_and_agents(self, agent_1, agent_2):
        self.GAME.reset()
        agent_1.MCTS.reset()
        agent_2.MCTS.reset()

    def _prepare_second_agent(self, agent_1):
        agent_2 = AlphaZeroAgent(
            player_number=-1,
            version=agent_1.VERSION,
            seconds_per_move=SCNDS_PER_MOVE_TRAINING,
            game=self.GAME,
            name_for_saving=self.NAME_FOR_SAVING
        )
        return agent_2

    def _compute_search_probabilities(self, agent: AlphaZeroAgent) -> np.array:
        search_probabilities = np.zeros(self.GAME.NO_ACTIONS)
        for child in agent.MCTS.current_root.children:
            action = child.action_before_state
            prob = child.visit_count / child.father.visit_count
            search_probabilities[action] = prob
        return search_probabilities

    def _prepare_nn_input(self, states_so_far, current_player):
        list_of_3_dimensional_states = []
        # last NR_BOARD_STATES_SAVED positions
        if len(states_so_far) >= NR_BOARD_STATES_SAVED:
            for state in states_so_far[-NR_BOARD_STATES_SAVED:]:
                list_of_3_dimensional_states.append(self._state_to_3_dimensional_state(state, current_player))
        else:
            # fill with empty states (just zeros)
            for i in range(NR_BOARD_STATES_SAVED - len(states_so_far)):
                empty_state = np.zeros(self.GAME.STATE_SHAPE)
                list_of_3_dimensional_states.append(self._state_to_3_dimensional_state(empty_state, current_player))
            for state in states_so_far:
                list_of_3_dimensional_states.append(self._state_to_3_dimensional_state(state, current_player))
        nn_input = np.array(list_of_3_dimensional_states)
        nn_input = nn_input.reshape(
            -1,
            3 * NR_BOARD_STATES_SAVED,
            *self.GAME.STATE_SHAPE)
        return nn_input

    def _state_to_3_dimensional_state(self, state, current_player):
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range(len(state)):
            layer_1.append([])
            layer_2.append([])
            layer_3.append([])
            for j in range(len(state[i])):
                if state[i][j] == 1:
                    layer_1[i].append(1)
                    layer_2[i].append(0)
                if state[i][j] == -1:
                    layer_1[i].append(0)
                    layer_2[i].append(1)
                if state[i][j] == 0:
                    layer_1[i].append(0)
                    layer_2[i].append(0)
                if current_player == 1:
                    layer_3[i].append(1)
                else:
                    layer_3[i].append(0)
        return [layer_1, layer_2, layer_3]
